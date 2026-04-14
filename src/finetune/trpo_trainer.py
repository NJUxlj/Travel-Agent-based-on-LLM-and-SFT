import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader  
import torch.nn.functional as F  
import numpy as np
from typing import Optional, Tuple, Dict, List
import logging
from collections import deque
import copy
import math
import numpy as np  
import copy
from datasets import load_dataset, load_from_disk  

from transformers import (  
    AutoTokenizer,  
    TrainingArguments,  
    Trainer,  
    TrainerCallback,  
    AutoModelForSequenceClassification,
)

from datasets import DatasetDict
from tqdm import tqdm
from pathlib import Path
import os, sys
sys.path.append(Path(__file__).parent.parent)
from models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from configs.config import (
    REWARD_MODEL_PATH, 
    MODEL_PATH, 
    SFT_MODEL_PATH, 
    TRPO_MODEL_PATH,
    PPO_MODEL_PATH, 
    DPO_DATA_PATH,
    CACHED_DPO_DATA_PATH,
    CACHED_PPO_DATA_PATH,
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TRPODataset(Dataset):  
    def __init__(self, tokenized_data):  
        self.data = tokenized_data  
        
    def __len__(self):  
        return len(self.data["input_ids"])  
    
    def __getitem__(self, idx):  
        return {  
            "input_ids": self.data["input_ids"][idx],  
            "attention_mask": self.data["attention_mask"][idx],  
            "response_ids": self.data["response_ids"][idx],  
            "old_log_probs": self.data["old_log_probs"][idx],  
        }  


class TRPOTrainer:
    '''
    Trust Region Policy Optimization

    '''

    def __init__(
        self,  
        output_dir: str = TRPO_MODEL_PATH,  
        dataset_path: str = DPO_DATA_PATH,  
        cached_data_path:str = CACHED_PPO_DATA_PATH,
        model_name: str = MODEL_PATH,  # 基础模型名称
        reward_model_name:str = REWARD_MODEL_PATH,
        reference_model_name: str = SFT_MODEL_PATH,  # 参考模型应该使用SFT模型
        clip_epsilon: float = 0.2,     # PPO裁剪参数，限制策略更新幅度
        gamma: float = 0.99,           # 折扣因子，控制未来奖励的重要性  
        gae_lambda: float = 0.95,      # GAE λ参数，平衡偏差和方差 
        ppo_epochs: int = 3,            # 每批数据的PPO更新次数 
        lr_actor: float = 5e-5,         # Actor学习率
        lr_critic: float = 5e-5,        # Critic学习率
        max_seq_length: int = 1024,  
        kl_coef: float = 0.1,          # KL散度系数
        vf_coef: float = 0.5,          # 价值损失系数
        entropy_coef: float = 0.01     # 熵奖励系数
    ):
        self.output_dir = output_dir  
        
        self.dataset_path = dataset_path
        self.cached_data_path = cached_data_path
        self.max_seq_length = max_seq_length
        
        self.clip_epsilon = clip_epsilon  
        self.gamma = gamma  
        self.gae_lambda = gae_lambda  
        self.ppo_epochs = ppo_epochs
        self.kl_coef = kl_coef
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef

        # 初始化模型和tokenizer  
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  
        self.tokenizer.pad_token = self.tokenizer.eos_token  
        
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        
        # 创建主策略模型（Actor）  - 这是我们要优化的模型 
        self.model = Qwen2ForCausalLM.from_pretrained(  
            model_name,  
            device_map="auto",  
            trust_remote_code=True  
        )  
        
        self.device  = self.model.device
        
        # 创建参考策略模型 - 用于KL散度计算，防止策略偏离太远  
        # 应该使用SFT模型作为参考模型，而不是与当前模型相同
        self.ref_model = Qwen2ForCausalLM.from_pretrained(  
            reference_model_name if reference_model_name else model_name,  
            device_map="auto",
            trust_remote_code=True  
        ).to(self.device)
        # 固定参考模型参数
        for param in self.ref_model.parameters():
            param.requires_grad = False

            
            
            
        # 创建奖励模型并设置为评估模式
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name,
            trust_remote_code=True
        ).to(self.device)
        # 设置为评估模式，因为我们只使用它来计算奖励而不训练它
        self.reward_model.eval()



    def _load_cached_dataset(self, dataset_path=CACHED_PPO_DATA_PATH):
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path, exist_ok=True)
        try:
            tokenized_data = load_from_disk(dataset_path)
            print("从缓存加载PPO数据集成功")
            train_data = tokenized_data['train']
            eval_data = tokenized_data['validation']
            # 确保数据格式正确
            train_data.set_format(type="torch")
            eval_data.set_format(type="torch")
            return TRPODataset(train_data), TRPODataset(eval_data)
        except Exception as e:
            print(f"加载缓存的PPO数据集失败: {e}, 将重新预处理数据")
            trpo_train_data, trpo_eval_data = self._prepare_dataset(self.dataset_path, self.max_seq_length)
            return trpo_train_data, trpo_eval_data

    def _prepare_dataset(self, dataset_path, max_length):  
        """数据预处理流程"""
        dataset = load_dataset(dataset_path)
        train_dataset = load_dataset(dataset_path, split='train').select(range(500))  
        
        if "validation" in dataset:
            eval_dataset = load_dataset(dataset_path, split='validation').select(range(500)) 
        else:
            eval_dataset = load_dataset(dataset_path, split='train').select(range(500, 1000))  
            
        def process_fn(samples):  
            batch = {"input_ids": [], "attention_mask": [],  
                    "response_ids": [], "old_log_probs": []}  
            
            for prompt, chosen in zip(samples["prompt"], samples["chosen"]):  
                # 生成完整prompt  
                full_prompt = f"Instruction: {prompt}\nResponse: {chosen}"  
                
                # Tokenize输入  
                tokens = self.tokenizer(  
                    full_prompt,  
                    max_length=max_length,  
                    padding="max_length",  
                    truncation=True,  
                    return_tensors="pt"  
                ).to(self.device)
                
                # 计算旧策略概率  (使用多项式采样替代argmax)
                with torch.no_grad():
                    logits = self.model.forward(**tokens).logits.to(self.device)
                    log_probs = F.log_softmax(logits, dim=-1)  
                    
                    # 多项式采样 - 向量化实现，提高效率
                    probs = torch.softmax(logits, dim=-1)  # shape: (batch_size, seq_len, vocab_size)
                    # 使用multinomial一次采样所有位置，避免循环
                    response_ids = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)  # (batch_size * seq_len, 1)
                    response_ids = response_ids.view(probs.size(0), probs.size(1))  # (batch_size, seq_len)
                    
                    # 只收集生成部分的log_prob
                    old_log_probs = torch.gather(  
                        log_probs, -1, response_ids.unsqueeze(-1)    # 把整个序列的logprobs都收集起来了，而不是仅仅收集action的部分
                    ).squeeze(-1)    # shape = (batch_size, seq_len)
                
                batch["input_ids"].append(tokens["input_ids"][0])  
                batch["attention_mask"].append(tokens["attention_mask"][0])  
                batch["response_ids"].append(response_ids[0])  
                batch["old_log_probs"].append(old_log_probs[0])  
            
            return batch  
        
        train_data = train_dataset.map(  
            process_fn,  
            batched=True,  
            num_proc=1,  
            remove_columns=train_dataset.column_names  
        )  
        
        val_data = eval_dataset.map(
            process_fn,
            batched=True,
            num_proc=1,
            remove_columns=eval_dataset.column_names
        )
        
        tokenized_data = DatasetDict({
            'train': train_data,
            'validation': val_data
        })
        
        
        # 保存预处理好的数据集到本地
        if not os.path.exists(CACHED_PPO_DATA_PATH):
            os.makedirs(CACHED_PPO_DATA_PATH, exist_ok=True)
        tokenized_data.save_to_disk(CACHED_PPO_DATA_PATH)
        
        
        train_data.set_format(type="torch")
        val_data.set_format(type="torch")
        
        return TRPODataset(train_data), TRPODataset(val_data)
    
    def compute_rewards(self, input_ids, attention_mask, response_ids):
        """
        计算奖励，结合模型输出和参考模型的KL散度
        
        参数:
            input_ids: 输入序列的token id
            attention_mask: 注意力掩码
            response_ids: 生成的响应序列
            
        返回:
            rewards: 计算得到的奖励
        """
        # 使用奖励模型计算奖励
        with torch.no_grad():
            # 准备奖励模型的输入
            reward_inputs = self.reward_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            reward_tokens = self.reward_tokenizer(
                reward_inputs,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            ).to(self.device)
            
            # 获取奖励模型的输出
            reward_outputs = self.reward_model(**reward_tokens)
            rewards = reward_outputs.logits.squeeze(-1)
            
            # 计算当前模型与参考模型之间的KL散度
            current_logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            current_probs = F.softmax(current_logits, dim=-1)
            
            with torch.no_grad():
                ref_logits = self.ref_model(input_ids=input_ids, attention_mask=attention_mask).logits
                ref_probs = F.softmax(ref_logits, dim=-1)
            
            # 计算KL散度
            kl_div = F.kl_div(
                F.log_softmax(current_logits, dim=-1),
                ref_probs,
                reduction='none'
            ).sum(dim=-1).mean(dim=1)  # 对每个样本计算平均KL散度
            
            # 结合奖励和KL散度惩罚
            rewards = rewards - self.kl_coef * kl_div
        
        return rewards
    
    def compute_advantages(self, rewards, masks, values):
        """
        计算广义优势估计(GAE)
        
        GAE公式: A_t = δ_t + γλA_{t+1}
        其中: δ_t = r_t + γV(s_{t+1}) - V(s_t) (TD误差)
        
        参数:
            rewards: 奖励序列
            masks: 掩码，标记有效token位置
            values: 价值估计序列
            
        返回:
            advantages: 计算得到的优势估计
        """
        # 初始化优势估计数组
        advantages = torch.zeros_like(values).to(self.device)
        last_gae_lam = 0
        
        # 从后向前计算GAE
        for t in reversed(range(values.size(1))):
            # 对于最后一个时间步，下一个状态的价值为0
            if t == values.size(1) - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]
            
            # 计算TD误差
            '''
            masks[:, t] （第t个时间步的掩码）有两个重要作用：
            
            1. 在TD误差计算中 
                - 这里 masks[:, t] 用于确保只有有效token（非padding）的未来价值被考虑。
                - 对于padding token， masks[:, t]=0 ，这样就会将 gamma * next_value * masks[:, t] 项置零，只保留当前奖励减去当前价值估计。
            
            2. 在递归计算GAE时 ：
                - last_gae_lam = delta + gamma * lam * masks[:, t] * last_gae_lam
                - 等价于 A(S_t) = delta + gamma * lam * mask[:, t] * A(S_{t+1})
                这里 masks[:, t] 用于控制是否将后续时间步的优势估计回传。对于padding token，后续时间步的优势不应影响当前步，因此将其置零。
            '''
            delta = rewards[:, t] + self.gamma * next_value * masks[:, t] - values[:, t]
            
            # 递归计算GAE
            last_gae_lam = delta + self.gamma * self.gae_lambda * masks[:, t] * last_gae_lam
            advantages[:, t] = last_gae_lam
        
        # 标准化优势估计
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def get_flat_params(self, model):
        """
        获取模型参数的扁平化表示
        
        参数:
            model: 要获取参数的模型
            
        返回:
            flat_params: 扁平化的参数向量
        """
        params = []
        for param in model.parameters():
            if param.requires_grad:
                params.append(param.view(-1))
        return torch.cat(params)
    
    def set_flat_params(self, model, flat_params):
        """
        将扁平化的参数向量设置回模型
        
        参数:
            model: 要设置参数的模型
            flat_params: 扁平化的参数向量
        """
        idx = 0
        for param in model.parameters():
            if param.requires_grad:
                param_size = param.numel()
                param.data.copy_(flat_params[idx:idx+param_size].view(param.size()))
                idx += param_size
    
    def fisher_vector_product(self, x, states, attention_mask, old_log_probs, damping=1e-2):
        """
        计算Fisher信息矩阵与向量的乘积
        
        Fisher信息矩阵通常很大，无法直接存储，因此使用Hessian向量积的方式计算
        
        参数:
            x: 输入向量
            states: 状态序列
            attention_mask: 注意力掩码
            old_log_probs: 旧策略的对数概率
            damping: 阻尼系数，用于数值稳定性
            
        返回:
            Fx: Fisher矩阵与x的乘积
        """
        # 计算KL散度的梯度
        self.model.zero_grad()
        
        # 前向传播
        logits = self.model(input_ids=states, attention_mask=attention_mask).logits
        new_log_probs = F.log_softmax(logits, dim=-1)
        
        # 计算KL散度
        kl = F.kl_div(
            new_log_probs,
            F.softmax(self.model(input_ids=states, attention_mask=attention_mask).logits.detach(), dim=-1),
            reduction='sum'
        )
        
        # 计算梯度
        grads = torch.autograd.grad(kl, self.model.parameters(), create_graph=True)
        flat_grads = torch.cat([g.view(-1) for g in grads if g is not None])
        
        # 计算梯度与向量x的点积
        grads_x = torch.dot(flat_grads, x)
        
        # 计算Hessian向量积
        hessian_vector_product = torch.autograd.grad(grads_x, self.model.parameters())
        flat_hessian_vector_product = torch.cat([g.contiguous().view(-1) for g in hessian_vector_product if g is not None])
        
        # 添加阻尼项以确保数值稳定性
        return flat_hessian_vector_product + damping * x
    
    def conjugate_gradient(self, Ax, b, max_iter=10, residual_tol=1e-10):
        """
        共轭梯度法求解Ax = b
        
        参数:
            Ax: 函数，计算A与任意向量的乘积
            b: 等式右侧的向量
            max_iter: 最大迭代次数
            residual_tol: 残差容忍度
            
        返回:
            x: 近似解
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        
        for i in range(max_iter):
            # 计算Ap
            Ap = Ax(p)
            
            # 计算步长
            alpha = rdotr / torch.dot(p, Ap)
            
            # 更新解和残差
            x = x + alpha * p
            r = r - alpha * Ap
            
            # 计算新的残差点积
            new_rdotr = torch.dot(r, r)
            
            # 检查收敛条件
            if torch.sqrt(new_rdotr) < residual_tol:
                break
            
            # 更新搜索方向
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        
        return x
    
    def line_search(self, states, attention_mask, response_ids, old_log_probs, advantages, full_step, expected_improvement, max_backtracks=10):
        """
        线搜索以找到满足信任区域约束的最优步长
        
        参数:
            states: 状态序列
            attention_mask: 注意力掩码
            response_ids: 响应序列
            old_log_probs: 旧策略的对数概率
            advantages: 优势估计
            full_step: 完整的更新步长
            expected_improvement: 预期的改进
            max_backtracks: 最大回溯次数
            
        返回:
            success: 是否找到有效的更新步长
            new_params: 更新后的参数
        """
        # 保存当前参数
        current_params = self.get_flat_params(self.model)
        
        # 计算当前目标函数值
        def compute_surrogate_loss(params):
            self.set_flat_params(self.model, params)
            
            # 计算新策略的对数概率
            logits = self.model(input_ids=states, attention_mask=attention_mask).logits
            new_log_probs = F.log_softmax(logits, dim=-1)
            
            # 只关注响应部分的概率
            action_log_probs = torch.gather(
                new_log_probs, -1, response_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            # 计算概率比
            ratio = torch.exp(action_log_probs - old_log_probs)
            
            # 计算替代损失
            surrogate_loss = -(ratio * advantages).mean()
            
            return surrogate_loss
        
        # 初始损失
        current_loss = compute_surrogate_loss(current_params)
        
        # 尝试不同的步长因子
        for i in range(max_backtracks):
            # 计算当前步长因子
            step_frac = 0.5 ** i
            
            # 计算新参数
            new_params = current_params + step_frac * full_step
            
            # 计算新损失
            new_loss = compute_surrogate_loss(new_params)
            
            # 计算实际改进
            actual_improvement = current_loss - new_loss
            
            # 检查是否满足条件：实际改进大于预期改进的一小部分，且实际改进为正
            if actual_improvement > 0 and actual_improvement > 0.1 * step_frac * expected_improvement:
                return True, new_params
        
        # 如果没有找到合适的步长，返回原始参数
        return False, current_params
    
    def trpo_step(self, states, attention_mask, response_ids, old_log_probs, advantages, values, max_kl=0.01):
        """
        执行TRPO策略更新步骤
        
        TRPO核心算法步骤：
        1. 计算策略梯度
        2. 使用共轭梯度法求解自然梯度
        3. 使用线搜索找到满足信任区域约束的最优步长
        4. 更新策略参数
        
        参数:
            states: 状态序列
            attention_mask: 注意力掩码
            response_ids: 响应序列
            old_log_probs: 旧策略的对数概率
            advantages: 优势估计
            values: 价值估计
            max_kl: 最大KL散度，定义信任区域的大小
        """
        # 1. 计算策略梯度
        self.model.zero_grad()
        
        # 计算新策略的对数概率
        logits = self.model(input_ids=states, attention_mask=attention_mask).logits
        new_log_probs = F.log_softmax(logits, dim=-1)
        
        # 获取动作的对数概率
        action_log_probs = torch.gather(
            new_log_probs, -1, response_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # 计算概率比
        ratio = torch.exp(action_log_probs - old_log_probs)
        
        # 计算替代损失
        surrogate_loss = -(ratio * advantages).mean()
        
        # 计算策略梯度
        grads = torch.autograd.grad(surrogate_loss, self.model.parameters())
        flat_grads = torch.cat([g.contiguous().view(-1) for g in grads if g is not None])
        
        # 2. 使用共轭梯度法求解自然梯度
        # 定义Fisher向量积函数
        def Fvp(v):
            return self.fisher_vector_product(v, states, attention_mask, old_log_probs)
        
        # 使用共轭梯度法求解 Fx = g
        step_direction = self.conjugate_gradient(Fvp, flat_grads)
        
        # 计算自然梯度步长
        # 确保步长在信任区域内: (1/2) * x^T F x ≤ max_kl
        shs = 0.5 * torch.dot(step_direction, Fvp(step_direction))
        lm = torch.sqrt(shs / max_kl)
        full_step = step_direction / lm
        
        # 计算预期改进
        expected_improvement = torch.dot(flat_grads, full_step)
        
        # 3. 使用线搜索找到满足信任区域约束的最优步长
        success, new_params = self.line_search(
            states, attention_mask, response_ids, old_log_probs, advantages,
            full_step, expected_improvement
        )
        
        # 4. 更新策略参数
        self.set_flat_params(self.model, new_params)
        
        return success
    
    def update_critic(self, states, attention_mask, returns, critic_optimizer):
        """
        更新价值函数估计器（Critic）
        
        参数:
            states: 状态序列
            attention_mask: 注意力掩码
            returns: 实际回报
            critic_optimizer: 价值函数优化器
            
        返回:
            value_loss: 价值损失
        """
        # 计算价值估计
        # 注意：在实际应用中，可能需要一个单独的价值网络
        # 这里简化处理，使用策略网络的输出作为价值估计
        with torch.no_grad():
            logits = self.model(input_ids=states, attention_mask=attention_mask).logits
        
        # 简化的价值估计（在实际应用中应该使用单独的价值网络）
        # 这里仅作为示例，实际实现需要根据模型架构进行调整
        values = logits.mean(dim=-1)  # 简化处理
        
        # 计算价值损失
        value_loss = F.mse_loss(values, returns)
        
        # 反向传播和优化
        critic_optimizer.zero_grad()
        value_loss.backward()
        critic_optimizer.step()
        
        return value_loss.item()
    
    def train(self, epochs=10, batch_size=4):
        """
        TRPO训练主循环
        
        参数:
            epochs: 训练轮数
            batch_size: 批次大小
        """
        # 加载数据集
        train_data, eval_data = self._load_cached_dataset()
        
        # 创建数据加载器
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_data, batch_size=batch_size)
        
        # 初始化优化器（仅用于Critic更新）
        critic_optimizer = optim.Adam(self.model.parameters(), lr=self.lr_critic)
        
        # 训练循环
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # 训练模式
            self.model.train()
            
            epoch_rewards = []
            epoch_value_losses = []
            successful_updates = 0
            total_updates = 0
            
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
                # 将数据移至设备
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                response_ids = batch["response_ids"].to(self.device)
                old_log_probs = batch["old_log_probs"].to(self.device)
                
                # 计算奖励
                rewards = self.compute_rewards(input_ids, attention_mask, response_ids)
                
                # 扩展奖励到序列长度
                expanded_rewards = rewards.unsqueeze(1).expand(-1, input_ids.size(1))
                
                # 计算价值估计（简化处理）
                with torch.no_grad():
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                    values = logits.mean(dim=-1)  # 简化处理
                
                # 计算广义优势估计
                advantages = self.compute_advantages(expanded_rewards, attention_mask, values)
                
                # 计算回报
                returns = advantages + values
                
                # 执行TRPO策略更新
                success = self.trpo_step(
                    input_ids, attention_mask, response_ids, old_log_probs, advantages, values
                )
                
                # 更新计数器
                total_updates += 1
                if success:
                    successful_updates += 1
                
                # 更新Critic
                value_loss = self.update_critic(input_ids, attention_mask, returns, critic_optimizer)
                
                # 记录指标
                epoch_rewards.append(rewards.mean().item())
                epoch_value_losses.append(value_loss)
            
            # 计算平均指标
            avg_reward = np.mean(epoch_rewards)
            avg_value_loss = np.mean(epoch_value_losses)
            success_rate = successful_updates / total_updates
            
            logger.info(f"Epoch {epoch+1} Stats:")
            logger.info(f"  Average Reward: {avg_reward:.4f}")
            logger.info(f"  Average Value Loss: {avg_value_loss:.4f}")
            logger.info(f"  Update Success Rate: {success_rate:.4f}")
            
            # 保存模型
            if (epoch + 1) % 2 == 0:
                save_path = os.path.join(self.output_dir, f"trpo_model_epoch_{epoch+1}")
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                logger.info(f"Model saved to {save_path}")
            
            # 评估
            self.evaluate(eval_loader, epoch)
        
        # 保存最终模型
        final_save_path = os.path.join(self.output_dir, "trpo_model_final")
        self.model.save_pretrained(final_save_path)
        self.tokenizer.save_pretrained(final_save_path)
        logger.info(f"Final model saved to {final_save_path}")
    
    def evaluate(self, eval_loader, epoch):
        """
        评估模型性能
        
        参数:
            eval_loader: 评估数据加载器
            epoch: 当前轮数
        """
        self.model.eval()
        
        eval_rewards = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Evaluating Epoch {epoch+1}"):
                # 将数据移至设备
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                response_ids = batch["response_ids"].to(self.device)
                
                # 计算奖励
                rewards = self.compute_rewards(input_ids, attention_mask, response_ids)
                eval_rewards.append(rewards.mean().item())
        
        # 计算平均奖励
        avg_eval_reward = np.mean(eval_rewards)
        
        logger.info(f"Evaluation Results for Epoch {epoch+1}:")
        logger.info(f"  Average Reward: {avg_eval_reward:.4f}")
        
        return avg_eval_reward


# 示例用法
if __name__ == "__main__":
    # 创建TRPO训练器实例
    trpo_trainer = TRPOTrainer(
        output_dir=TRPO_MODEL_PATH,
        dataset_path=DPO_DATA_PATH,
        cached_data_path=CACHED_PPO_DATA_PATH,
        model_name=MODEL_PATH,
        reward_model_name=REWARD_MODEL_PATH,
        reference_model_name=SFT_MODEL_PATH,
        clip_epsilon=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_epochs=3,
        lr_actor=5e-5,
        lr_critic=5e-5,
        max_seq_length=1024,
        kl_coef=0.1,
        vf_coef=0.5,
        entropy_coef=0.01
    )
    
    # 开始训练
    logger.info("Starting TRPO training...")
    trpo_trainer.train(epochs=5, batch_size=4)
    logger.info("TRPO training completed!")