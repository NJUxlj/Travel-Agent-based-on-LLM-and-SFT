from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


from typing import List, Tuple, Union, Dict, Optional

from src.configs.config import MODEL_PATH, REWARD_MODEL_PATH



# 构建dataset
class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer: AutoTokenizer, apply_chat_template=False):
        self.prompts = prompts
        self.tokenizer:AutoTokenizer = tokenizer
        
        self.final_prompts = []
        
        for prompt in prompts:
            if apply_chat_template:
                content = [{"role":"user", "content":prompt}]    
                prompt = self.tokenizer.apply_chat_template(content, tokenize = False, add_generation_prompt = True)
            else:
                prompt = self.tokenizer.bos_token + prompt
                
            self.final_prompts.append(prompt)
        
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.final_prompts[idx]
    
    
    



# 价值（评论家）模型，用于预测每一步（生成token）的动作产生的收益，使用演员模型进行初始化，并外加一个回归头，输出shape为：(batch_size, seq_len， 1)
class Critic(nn.Module):
    '''
    # 价值（评论家）模型：
    #   - 用于预测每一步（生成token）的动作产生的收益，使用演员模型进行初始化，并外加一个回归头，输出shape为：(batch_size, seq_len， 1)
    #   - 通过策略模型初始化得到
    
    '''
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model  # 可以看做一个 encoder
        self.base_model.eval()  # 冻结骨干部分
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask, num_actions):
        '''
        num_asctions: 模型新推理了多少token
        
        return values: shape = (batch_size, num_actions)
        '''
        
        hidden_states = self.base_model.forward(
            input_ids,
            attention_mask = attention_mask
        ).last_hidden_state  # shape = (batch_size, seq_len, hidden_size)
        
        value_model_output = self.value_head.forward(hidden_states)  # shape = (batch_size, seq_len, 1)
        
        values = value_model_output.squeeze(-1)[:, -num_actions:] # 只取策略做出的actions（response）的部分
        
        return values
    
    






def compute_policy_loss(log_probs:torch.Tensor, old_log_probs:torch.Tensor, advantages, action_mask = None, clip_eps = 0.2):
    '''
    advantage: 
        优势函数，用于衡量当前策略相对于旧策略的优势。
        advantage.shape = (batch_size, seq_len)
        
    log_probs:
        策略模型输出的logits
        log_probs.shape = (batch_size, seq_len)
        
    old_log_probs:
        旧策略模型输出的logits
        old_log_probs.shape = (batch_size, seq_len)
        
    action_mask:
        用于指示哪些动作是有效的，哪些动作是无效的。
        action_mask.shape = (batch_size, seq_len)
    '''
    ratio = (log_probs - old_log_probs).exp()  # A/B = exp[log(A/B)] = exp[logA - LogB] 
    
    # ratio.shape = (batch_size, seq_len)
    
    surrogate1 = ratio * advantages
    surrogate2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    
    loss = - torch.min(surrogate1, surrogate2) # shape =  (bsz, seq_len)
     
    if action_mask is None:
        return loss.mean(-1).mean()  # 先对序列维度求平均，再对批次维度求平均
    else:
        return ((loss*action_mask).sum(-1) / action_mask.sum(-1)).mean()




def compute_value_loss(values, old_values, returns, action_mask = None, clip_eps:float = None):
    '''
    计算价值模型的损失函数（Value Loss），用于优化评论家（Critic）模型。
    
    参数:
        values (torch.Tensor): 当前价值模型预测的状态价值，shape=(batch_size, seq_len), Critic模型预测的状态价值（V(s)），表示在当前策略下从状态s开始的预期回报
        old_values (torch.Tensor): 旧价值模型预测的状态价值，shape=(batch_size, seq_len)
        returns (torch.Tensor): 实际观察到的回报（return），shape=(batch_size, seq_len), 实际观察到的折扣回报（G_t）
        action_mask (torch.Tensor, optional): 动作掩码，指示哪些位置是有效动作，shape=(batch_size, seq_len)
        clip_eps (float, optional): 用于限制价值更新的裁剪系数。如果为None则不使用裁剪
        
    返回:
        torch.Tensor: 计算得到的价值损失标量值
        
    损失函数：
        loss = (values - returns) ** 2
        等价于  L = (V(s) - G_t)²
        
        本质：
            是价值函数（Critic）训练中的均方误差（MSE）损失计算
        
        作用：  
            1. 这是典型的时序差分（Temporal Difference）学习中的价值函数拟合
            2. 目标是最小化预测价值与实际回报之间的平方误差
            3. 通过最小化这个损失，Critic模型逐渐学会更准确地预测每个状态的期望回报
        
        其中：
            values：Critic模型预测的状态价值（V(s)），表示在当前策略下从状态s开始的预期回报
            returns：实际观察到的折扣回报（G_t），通过蒙特卡洛方法或TD(λ)等方法计算得到
        
    实现细节:
        1. 当clip_eps不为None时，使用PPO的裁剪机制防止价值函数更新过大:
           - 计算裁剪后的价值预测: values_clipped = old_values + clip(values - old_values)
           - 取裁剪和非裁剪损失中的较大值: loss = max((values_clipped-returns)^2, (values-returns)^2)
        2. 当clip_eps为None时，直接计算MSE损失: loss = (values - returns)^2
        3. 如果有action_mask，则只计算有效位置的损失
        4. 最终对序列维度和批次维度求平均得到标量损失值
        
    数学公式:
        clipped_loss = (values_clipped - returns)^2
        unclipped_loss = (values - returns)^2
        loss = max(clipped_loss, unclipped_loss)  # 当使用clip时
        或
        loss = unclipped_loss  # 当不使用clip时
    '''
    if clip_eps is not None:
        values_clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        surrogate1 = (values_clipped- returns)**2
        surrogate2 = (values - returns) ** 2
        
        loss = torch.max(surrogate1, surrogate2)
    
    else:
        loss = (values - returns) ** 2
    
    
    if action_mask is None:
        return loss.mean(-1).mean()  # 先对序列维度求平均，再对批次维度求平均

    else:
        return ((loss*action_mask).sum(-1) / action_mask.sum(-1)).mean()



    

def compute_entropy(log_probs: torch.Tensor, action_mask: Optional[torch.Tensor] = None):
    '''
    计算策略的熵，用于鼓励探索
    
    参数:
        log_probs: 策略模型输出的对数概率，shape=(batch_size, num_actions)
        action_mask: 动作掩码，指示哪些位置是有效动作，shape=(batch_size, num_actions)
        
    返回:
        torch.Tensor: 计算得到的熵标量值
        
    公式:
        entropy = -sum(exp(log_probs) * log_probs)

    注意：
        1. 熵计算是沿着动作维度(num_actions)进行的
        2. 最终返回的是整个batch的平均熵值。
    '''
    probs = log_probs.exp()
    per_token_entropy = - (probs * log_probs) # shape = (bsz, num_actions)
    entropy = per_token_entropy.sum(-1)  # shape = (bsz,)
    
    if action_mask is None:
        return entropy.mean()
    else:
        return ((per_token_entropy * action_mask).sum(-1)).mean()
    
    
    
    
class ExperienceBuffer:
    '''
    经验池
    
    作用：
        存储我们采样到的轨迹：Trajectories
    '''
    def __init__(self, limit):
        self.limit = limit
        self.buffer = []  # 存储轨迹的列表，也就是实际上的轨迹数据集D

    
    def append(self, experiences:List["Experience"]):
        '''
        删除旧数据， 保留新数据
        
        作用：
            将experiences对象列表，转为batch这个字典列表
            
            再把batch添加到buffer里面
        '''
        batch = [{} for _ in range(len(experiences))]  
        
        keys = (
            "seqs",
            "action_log_probs",
            "values",  # Critic模型预测的状态价值（V(s)），shape = (batch_size, seq_len)
            "returns", # 实际观察到的折扣回报（G_t），shape = (batch_size, seq_len)
            "advantages",
            "attention_mask",
            "action_mask",
            "num_actions" # 模型新推理了多少token
        )
        
        for key in keys:
            for i, experience in enumerate(experiences):
                value = getattr(experience, key)
                batch[i][key] = value
        
        self.buffer.extend(batch)
        
        
        if len(self.buffer) >= self.limit:
            self.buffer = self.buffer[len(self.buffer)-self.limit:]
            
        
        
        
    def get_batches(self, batch_size):
        '''
        随机抽取一个批次
        '''
        batch = random.sample(self.buffer, batch_size)
        return batch

    
    
    
    def clear(self):
        self.buffer = []
        
    def __len__(self):
        return len(self.buffer)
    
    
    
    def __getitem__(self, index):
        return self.buffer[index]
    
    
    


@dataclass
class Samples:
    '''
    存储策略模型的输出
    '''
    seqs:torch.Tensor
    attention_mask:Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]  # shape = (batch_size, seq_len)
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    
    
@dataclass
class Experience:
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    reward: torch.Tensor
    response_length: torch.Tensor
    total_length: torch.Tensor
    num_actions: Union[int, torch.Tensor]
    kl: Optional[torch.Tensor] = None




def compute_approx_kl(
    log_probs: torch.Tensor,  # 策略模型输出的logits
    ref_log_probs: torch.Tensor, # 参考模型输出的logits
    action_mask: Optional[torch.Tensor] = None,
):
    '''
    log_probs.shape = (batch_size, seq_len)
    
    return log_ratio, shape = (batch_size, seq_len)
    '''
    log_ratio = log_probs.float() - ref_log_probs.float()  # log(A/B) = logA - logB
    
    if action_mask  is not None:
        log_ratio = log_ratio * action_mask
    
    return log_ratio





def get_advantages_and_returns(
    values: torch.Tensor,
    rewards: torch.Tensor,
    action_mask: torch.Tensor,
    gamma: float,
    lambd: float
    ):
    '''
    ### Args:
        values: 价值模型预测的状态价值（V(s)），shape = (batch_size, seq_len)
        rewards: 奖励模型预测的奖励（R_t），shape = (batch_size, seq_len)
        action_mask: 动作掩码，指示哪些位置是有效动作，shape=(batch_size, seq_len)
        gamma: 折扣因子
    
    
    
    ### 原理：
        # A(t) = R(t) + gam*V(t+1) - V(t)
        # gae:A(t) = R(t) + gam*V(t+1) - V(t) + gam*lam*A(t+1)
        # 最后一个时刻的未来优势和未来收益为0：A(T+1) = 0, V(T+1) = 0,  则A(T) = R(T) - V(T), 得出A(T)
        # A(T-1) = R(T-1) + gam*V(T) - V(T-1) + gam*lam*A(T) 知道A(T)可计算A(T-1) 依次类推
        # returns(t) = A(t) + V(t) = = R(t) + gam * (V(t+1) + lam * A(t+1))

        
        returns(t) = A(t) + V(t) 
            = [R(t) + gamma*V(t+1) - V(t)] + V(t)  # 代入普通优势函数定义
            = R(t) + gamma*V(t+1)  # 标准的回报公式
            
        returns(t) = A(t) + V(t) 
            = [R(t) + gamma*V(t+1) - V(t) + gam*lam*A(t+1)] + V(t)  # 代入广义优势函数定义
            = R(t) + gamma*V(t+1) + gam*lam*A(t+1)  # 回报公式
            
    ### Return: (advantages, returns)
        advantages: 优势函数，用于衡量当前策略相对于旧策略的优势。
            shape = (batch_size, seq_len)
            
        returns: shape=(batch_size, seq_len), 实际观察到的折扣回报（G_t）
            shape = (batch_size, seq_len)
            
    '''
    
    last_gae_lam = 0 # A_{T+1} 初始化为0， 因为最后一个时刻的未来优势和未来收益为0：A(T+1) = 0, V(T+1) = 0
    
    advantages_reversed = [] #  由于一开始我们从最后一步的优势 A_T = delta + gamma*lambda* A_{T+1} 开始计算， 因此最后需要把结果反转一下
    
    response_length = rewards.size(1)   # 一个轨迹中，最后的所有actions的长度 （或，所有 response tokens的长度） 
    
    
    if action_mask is not None:
        values = action_mask * values
        rewards = action_mask * rewards
        
    for t in reversed(range(response_length)):
    
        nextvalues = values[:, t+1] if t <response_length-1 else 0.0 # V(t+1)    shape = (batch_size)
        delta = rewards[:, t] + gamma * nextvalues - values[:, t] # delta = R(t) + gam*V(t+1) - V(t)     shape = (batch_size)
        
        last_gae_lam = delta + gamma * lambd * last_gae_lam # A(t) = delta + gam*lam*A(t+1) = R(t) + gam*V(t+1) - V(t) + gam*lam*A(t+1)       # shape  = (batch_size)

        advantages_reversed.append(last_gae_lam)   # List[torch(batch,) for _ in range(response_length)]
        
        
    advantages = torch.stack(advantages_reversed[::-1], dim=1)  # shape = (batch_size, response_length)
    
    returns = advantages + values  # 基本的预期回报公式

    return advantages.detach(), returns



def generate_samples(
    prompts,
    model,
    max_length,
    max_new_tokens,
    n_samples_per_prompt,
    micro_rollout_batch_size, # 轨迹数据集D中的一个 batch 的大小
)->List[Samples]:
    '''
    采样一整个数据集D的轨迹数据
    
    return List[Samples]

        where lene(Samples) == micro_rollout_batch_size    
    '''
    
    samples_list = []
    model.eval()
    all_prompts = sum([[prompt]*n_samples_per_prompt for prompt in prompts], []) # 返回值：一个列表，包含每个prompt重复n_samples_per_prompt次的结果
    
    for i in range(0, len(all_prompts), micro_rollout_batch_size):
        
        prompts = all_prompts[i: i+micro_rollout_batch_size]
        inputs = actor_tokenizer(prompts, padding = "max_length", max_length=max_length, truncation=True, return_tensors = "pt")
        input_ids = inputs['input_ids']
        seqs = model.generate(
            **inputs.to(device),
            max_new_tokens = max_new_tokens,
            eos_token_id = eos_token_id,
            pad_token_id = pad_token_id
        ) # 生成完整的轨迹
        
        # seqs.shape = (micro_rollout_batch_size, max_new_tokens + max_length)
        
        if seqs.size(1) >= max_new_tokens + max_length:
            seqs = seqs[:, :max_new_tokens + max_length]  # 截断
        else:
            seqs = torch.cat([seqs, torch.full((seqs.size(0), max_new_tokens+max_length-seqs.size(1)), fill_value=pad_token_id, device = seqs.device)], dim=1)   # 补全
        
        
        attention_mask = (seqs.ne(pad_token_id)).to(dtype = torch.long)
        ans = seqs[:, input_ids.size(1):]
        
        action_mask = (ans.ne(pad_token_id) & ans.ne(eos_token_id)).to(dtype = torch.long)  # 最后一个有效 action的位置在每一个 sequence中都是相同的
        
        
        samples = Samples(
            seqs = seqs,
            attention_mask = attention_mask,
            action_mask = action_mask,
            num_actions= action_mask.size(1),
            packed_seq_lens = None,
            response_length = action_mask.float().sum(dim=-1), 
            total_length =  attention_mask.float().sum(dim=-1)
            
        )
        
        samples_list.append(samples)

    return samples_list
    
    



def compute_rewards(kl, r, action_mask, kl_ctl, clip_reward_value):
    '''
    ### Args:
        kl: 策略模型输出的logits, shape  = (bsz, num_actions)
        r: 奖励模型输出的奖励（R_t），shape = (batch_size, )
        action_mask: 动作掩码，指示哪些位置是有效动作，shape=(batch_size, num_actions)
        kl_ctl: KL散度的控制系数(权重)
        clip_reward_value: 奖励裁剪阈值
        
    ### 最终返回的奖励tensor结构：
        大部分token的奖励=KL惩罚
        最后一个有效token的奖励=KL惩罚 + 裁剪后的奖励模型输出
        
    ### Return:
        rewards: 奖励，shape = (bsz, num_actions)
        
    ### 注意:
        这里，我们简化了每个token的奖励计算，仅仅认为他们是 -KL散度 (如果是序列中的最后一个token，还要加上奖励模型的预测分数)，
            但在实际应用中，我们会用更加细致的奖励模型来预测每个token的奖励。
    '''
    
    kl_divergence_estimate = -kl_ctl * kl   # 负号表示惩罚KL散度大的情况
    rewards = kl_divergence_estimate   # shape = (bsz, num_actions)    # 初始奖励=KL惩罚
    
    ends = action_mask.sum(1) + 1   # shape=(bsz,)
    
    if not isinstance(clip_reward_value, torch.Tensor):
        clip_reward_value = torch.tensor(clip_reward_value).to(r.device)
    # 对每个序列的最后一个token的奖励进行裁剪
    reward_clip = torch.clamp(r, -clip_reward_value, clip_reward_value)  # shape = (bsz,)
    
    # 将裁剪后的最终奖励加到每个序列的最后一个有效token上
    batch_size = r.size(0)
    for j in range(batch_size):
        rewards[j, :ends[j]][-1] += reward_clip[j, 0]  # 只在序列末尾加奖励
        
        
        
    return rewards
    
    




def generate_experiences(samples_list:List[Samples])->List[Experience]:
    
    actor_model.eval()
    ref_model.eval()
    reward_model.eval()
    critic_model.eval()
    
    experiences  =[]
    
    
    
    for samples in samples_list:
        seqs = samples.seqs  # seqs.shape = (micro_rollout_batch_size, max_new_tokens + max_length)
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        
        with torch.no_grad():
            # 计算策略模型输出 token 的概率
            output = actor_model(seqs, attention_mask = attention_mask)

    
            logits  = output.logits  # shape = (bsz, max_len + max_new_tokens, vocab_size)
            # 为什么取到-1：因为这是标准的"teacher forcing"做法，预测下一个token时不需要最后一个token的预测结果
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # shape = (bsz, max_len + max_new_tokens - 1, vocab_size)
            # 为什么从1开始取：因为要获取每个token预测下一个token的概率（即用前n-1个token预测第n个token）
            # gather操作：
            #     在最后一个维度（dim=-1，即vocab维度）进行索引
            #     效果：对于每个batch中的每个位置，选择实际生成的那个token对应的对数概率
            #     相当于：log_probs_labels[i,j] = log_probs[i,j,seqs[i,j]] , 但是 seqs[i,j] 装的实际上是 seqs[i,j+1]位置的token id
            log_probs_labels = log_probs.gather(dim=-1, index = seqs[:, 1:].unsqueeze(-1)) # shape = (bsz, max_len + max_new_tokens - 1, 1)
            action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]  # shape = (bsz, num_actions)
      
            # 计算参考模型输出 token的概率
            ref_output = ref_model(seqs, attention_mask = attention_mask)
            ref_logits = ref_output.logits
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
    
            ref_log_probs_labels = ref_log_probs.gather(dim=-1, index = seqs[:,1:].unsqueeze(-1)) # shape = (bsz, max_len + max_new_tokens - 1, 1)

            ref_action_log_probs = ref_log_probs_labels.squeeze(-1)[:, -num_actions:]
            
            # 计算价值
            
            value = critic_model.forward(seqs, attention_mask, num_actions).to(device) # shape = (bsz, num_actions)
            
            # 转换成文本
            seq_texts = actor_tokenizer.batch_decode(seqs, skip_special_tokens = True)
            
            reward_model_inputs  =  reward_tokenizer(seq_texts, return_tensors="pt", padding=True)

            r = reward_model(**reward_model_inputs.to(device)).logits # shape = (bsz, )  # 奖励模型的输出，相当于生成最后一个token的奖励分数（结果奖励模型）
            
            # 计算 kl 散度
            kl = compute_approx_kl(
                action_log_probs,
                ref_action_log_probs,
                action_mask = action_mask
            ).to(device) # shape = (bsz, num_actions)
            
            # 计算实际奖励
            
            
            rewards = compute_rewards(kl, r, action_mask, kl_ctl=0.1, clip_reward_value=0.2) # shape = (bsz, num_actions)
            
            # 计算优势和回报
            advantages, returns = get_advantages_and_returns(value, rewards, action_mask, gamma=0.1, lambd=0.2)

        # actor_model.train()
        # critic_model.train() 
        experiences.append(
            Experience(
                seqs,
                action_log_probs.detach(),
                value.detach(),
                returns.detach(),
                advantages.detach(),
                attention_mask,
                action_mask,
                r.detach(),
                samples.response_length,
                samples.total_length,
                num_actions,
                kl.detach()
            )
        )
    
    return experiences






@dataclass
class BufferItem:
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    num_actions: Union[int, torch.Tensor]
    
    

def collate_fn(batch):
    '''
    batch: List[Experience]  -> List[Samples]
    
    len(batch) == micro_train_batch_size == 2
    
    len(Samples) == micro_rollout_batch_size == 2
    
    一个 batch中包含了 micro_train_batch_size* micro_rollout_batch_size 个轨迹样本
    '''
    seqs:List[List] = []     # [ [seq1, seq2], [seq3, seq4] ]
    action_log_probs = []
    values = []
    returns = []
    advantages = []
    attention_mask = []
    action_mask = []
    
    for x in batch:
        seqs.append(x['seqs'])
        action_log_probs.append(x['action_log_probs'])
        values.append(x['values'])
        returns.append(x['returns'])
        advantages.append(x['advantages'])
        attention_mask.append(x['attention_mask'])
        action_mask.append(x['action_mask'])

    seqs = torch.cat(seqs, dim=0) # 在行方向纵向拼接两个序列成为一个矩阵
    action_log_probs = torch.cat(action_log_probs, dim=0)
    values = torch.cat(values, dim=0)
    returns = torch.cat(returns, dim=0)
    advantages = torch.cat(advantages, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    action_mask = torch.cat(action_mask, dim=0)
    
    return BufferItem(seqs, action_log_probs, values, returns, advantages, attention_mask, action_mask, action_mask.size(1))







def train_step(experience:Experience, steps):
    '''
    做一个mini-batch的训练
    '''
    
    actor_model.train()
    optimizer_actor.zero_grad()
    
    sequences = experience.seqs
    old_action_log_probs = experience.action_log_probs  # 含义： 原始的actor模型输出的logits
    advantages = experience.advantages  # shape = (batch_size, num_actions)
    num_actions = experience.num_actions
    attention_mask = experience.attention_mask
    action_mask = experience.action_mask
    old_values = experience.values
    returns = experience.returns
    
    
    
    logits = actor_model(
            sequences,
            attention_mask = attention_mask
        ).logits
    
    log_probs  = F.log_softmax(logits[:, :-1, :], dim =-1)  # shape = (batch_size, max_len + max_new_tokens - 1, vocab_size)
    log_probs_labels = log_probs.gather(dim=-1, index = sequences[:, 1:].unsqueeze(-1)) # shape = (batch_size, max_len + max_new_tokens - 1, 1)
    action_log_probs = log_probs_labels.squeeze(-1)[:,-num_actions:]  # shape = (batch_size, num_actions)

    # 计算策略损失
    policy_loss = compute_policy_loss(action_log_probs, old_action_log_probs, advantages, action_mask = action_mask)

    # 计算熵
    entropy = compute_entropy(action_log_probs, action_mask)  # 计算一个 mini-batch的平均entropy
    
    # 计算价值损失
    critic_model.train()
    optimizer_critic.zero_grad()
    values = critic_model.forward(sequences, attention_mask, num_actions)  # shape = (batch_size, num_actions)
    value_loss = compute_value_loss(values, old_values, returns, action_mask)
    

    # 合并3种损失
    total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy  # 系数可根据需要调整
    
    # 反向传播
    total_loss.backward()
    optimizer_actor.step()
    optimizer_critic.step()
    
    # policy_loss.backward()
    # optimizer_actor.step()
    
    # writer.add_scalar("policy_loss", policy_loss.item(), steps)
    # value_loss.backward()
    # optimizer_critic.step()
    # writer.add_scalar("value_loss", value_loss.item(), steps)
    
    # 记录日志
    writer.add_scalar("policy_loss", policy_loss.item(), steps)
    writer.add_scalar("value_loss", value_loss.item(), steps)
    writer.add_scalar("entropy", entropy.item(), steps)
    writer.add_scalar("total_loss", total_loss.item(), steps)

    # print(f"step:{steps} policy_loss:{policy_loss.item():.4f} value_loss:{value_loss.item():.4f}" )
    print(f"step:{steps} policy_loss:{policy_loss.item():.4f} value_loss:{value_loss.item():.4f} entropy:{entropy.item():.4f} total_loss:{total_loss.item():.4f}")
    
    
def train():
    # 初始化经验池, 经验池就是论文中的 轨迹数据集 D (Trajectories)
    buffer = ExperienceBuffer(limit=100)   # 先取 100 个 prompt， yong来生成轨迹
    
    steps = 0
    
    
    for episode in range(episodes):
        for rand_prompts in prompts_dataloader:
             # 生成样本（获取模型推理结果）
                # 采样一个 数据集 D 的轨迹
            samples:List[Samples] = generate_samples(rand_prompts, actor_model, max_length, max_new_tokens, n_samples_per_prompt, micro_rollout_batch_size)
            
             # 生成经验（获取优势、奖励、回报等）
            experiences = generate_experiences(samples) # 计算轨迹数据集D中的每条轨迹的奖励、回报、优势等
            
            buffer.append(experiences)  # buffer 只存储一个 数据集D的轨迹数据
            
            dataloader = DataLoader(buffer, batch_size = micro_train_batch_size, shuffle = True, collate_fn = collate_fn)
            
            torch.cuda.empty_cache()
            
            for epoch in range(max_epochs):  # 每个 数据集D 都要训练 max_epochs 轮
                for experience in dataloader:      # 每个 experience 都是一个 mini-batch
                    train_step(experience, steps)
                    steps+=1
            
            
            buffer.clear()
            torch.cuda.empty_cache()



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 一共迭代多少轮
    episodes = 3
    # 生成一次经验 (i.e., Trajectories dataset)，训练的轮数
    max_epochs = 5
    # 一次从提示词数据集中取多少条数据用于生成经验 
    rollout_batch_size = 8
    # 一次取多少条数据生成经验（生成经验需要多个模型推理，对显存要求高） # mini-batch
    micro_rollout_batch_size = 2
    # 一个提示词生成多少个样本
    n_samples_per_prompt = 2
    
    # 下面的prompt_dataset 可以看做是一个更大的数据集 D'，从中，我们每次获取一个子集 D
    
    # 轨迹数据集 D 的大小 =  rollout_batch_size *  n_samples_per_prompt
    
    # 每次从D中拿出 mini-batch(micro_rollout_batch_size, 或 micro_train_batch_size)条轨迹用来更新 actor_model (online policy)
    
    
    # 生成的最大长度，相当于最大动作数，数值越大，模型探索的可能性越多
    max_new_tokens = 50
    # 最大长度
    max_length = 256
    # 实际训练的batch_size大小，一次取多少条数据用于更新参数
    micro_train_batch_size = 2     # mini-batch
    # 记录日志
    writer = SummaryWriter('./runs')
    # 策略模型
    actor_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    # 参考模型
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    # 奖励模型, 如何使用： score = reward_model(**input_ids).logits  # shape = (bsz, )
    reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_PATH).to(device)
    
    
    # actor_tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct')
    actor_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)
    # 价值模型
    critic_model = Critic(actor_model.base_model).to(device)
    
    # 初始化优化器
    optimizer_actor = torch.optim.Adam(actor_model.parameters(), lr=0.00005)
    optimizer_critic = torch.optim.Adam(critic_model.parameters(), lr=0.00005)
    
    # 填充方式为左填充
    actor_tokenizer.padding_side = 'left'
    eos_token_id = actor_tokenizer.eos_token_id
    pad_token_id = actor_tokenizer.pad_token_id
    prompt_list = [
        '请问1+1等于多少？',
        'PowerShell，如何知道BIOS中的虚拟化是否已禁用',
        '为什么人们喜欢在水族馆里游泳，而不是在游泳池里？',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '为什么所有的镜子都是矩形的？',
        '我们在受感染的植物根部可以找到哪一种，臭氧还是金子？'
    ]
    prompts_dataset = PromptDataset(prompt_list, actor_tokenizer, apply_chat_template=True)
    prompts_dataloader = DataLoader(prompts_dataset, batch_size=rollout_batch_size, shuffle=True)
   
    train()