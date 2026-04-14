from typing import Dict, Optional
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    
)
from peft import get_peft_model, LoraConfig, TaskType
import bitsandbytes as bnb

import sys
sys.path.append("../../")  # 添加上级目录的上级目录到sys.path
from src.configs.config import MODEL_CONFIG

from src.utils.utils import (
    load_qwen_in_4bit,
    load_split_model,
    load_qwen
)

try:
    from src.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
except Exception as e:
    print("modeling_qwen2 导包出现问题， 应该是transformers版本过低：",str(e))
    

# from src.finetune.sft_trainer import SFTTrainer
# from src.finetune.dpo_trainer import DPOTrainer

from zhipuai import ZhipuAI
import os
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY")


class TravelAgent:

    def __init__(
        self,
        model_name: str = MODEL_CONFIG['model']['name'], 
        device_map: str = "auto",
        device: str = "cuda" if torch.cuda.is_available() else "cpu", 
        lora_config: Optional[Dict] = None,
        sft_trainer = None,
        dpo_trainer = None,
        use_bnb=False,
        use_lora = False,
        use_sft = False,
        use_dpo = False,
        use_ppo = False,
        use_grpo = False,
        use_api = False,
        ) -> tuple:
        """
        加载基础模型和分词器
        
        Args:
            model_name: 模型名称或路径
            device_map: 设备映射策略
        
        Returns:
            tuple: (model, tokenizer)
        """
        # 初始化基础配置  
        self.device = device  
        self.device_map = device_map
        self.model_name = model_name  
        self.use_bnb=use_bnb
        self.use_lora = use_lora
        self.use_sft = use_sft
        self.use_dpo = use_dpo
        self.use_ppo = use_ppo
        self.use_grpo = use_grpo
        self.use_api = use_api
        
        
        
        if self.use_api:
            # API 模式下不加载模型，避免内存问题
            self.model = None
            self.tokenizer = None
            print("使用 API 模式，跳过本地模型加载")
        
        else:
        
            # 默认LoRA配置  
            self.lora_config = {  
                "r": 8,  # LoRA秩  
                "lora_alpha": 32,  # LoRA alpha参数  
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],  # 需要训练的模块  
                "lora_dropout": 0.1,  
                "bias": "none",  
                "task_type": TaskType.CAUSAL_LM  
            }  if lora_config is None else lora_config
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            
            # 确保分词器具有pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model: Qwen2ForCausalLM = self._init_model().to(self.device)
            
            
            
            
            if self.use_sft:
                self.sft_trainer = sft_trainer
                
            if self.use_dpo:
                self.dpo_trainer = dpo_trainer
            
            
            torch.cuda.empty_cache()  
            
            
    def call_api_model(self, prompt):
        client = ZhipuAI(api_key=ZHIPU_API_KEY) 
        response = client.chat.completions.create(
            model="glm-4-flash",  # 填写需要调用的模型名称
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        response_text = response.choices[0].message.content
        return response_text
        
        
    
    def _init_model(self) -> AutoModelForCausalLM:  
        """初始化模型并应用LoRA配置"""  
        # 加载基础模型  
        # model = AutoModelForCausalLM.from_pretrained(  
        #     self.model_name,  
        #     trust_remote_code=True,
        #     torch_dtype=torch.float16,  
        #     load_in_8bit = True if self.use_bnb else False,
        #     # device_map=self.device_map  # 并行训练时， 不能使用自动设备映射
        # )
        
        
        if self.use_bnb:
            model = load_qwen_in_4bit(self.model_name)
        else:
            model = load_qwen(self.model_name)
        
        
        if self.use_lora:
        # 应用LoRA配置  
            peft_config = LoraConfig(  
                r=self.lora_config["r"],  
                lora_alpha=self.lora_config["lora_alpha"],  
                target_modules=self.lora_config["target_modules"],  
                lora_dropout=self.lora_config["lora_dropout"],  
                bias=self.lora_config["bias"],  
                task_type=self.lora_config["task_type"]  
            )  
            
            model = get_peft_model(model, peft_config)  
        
        return model  
    

    def generate_response(
        self,
        prompt:str,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        生成模型响应
        
        Args:

            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top-p采样参数
        
        Returns:
            str: 生成的响应文本
        """
        if self.use_api:
            # 使用 API 调用
            return self.call_api_model(prompt)
        
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成回复
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 移除prompt部分
        response = response[len(prompt):]
        
        return response.strip()
    
    
    
    
    def chat(self):  
        """简单聊天功能，交互式对话"""  
        
        history = [("You are a very help AI assistant who can help me plan a wonderful trip for a vacation",
                    "OK, I know you want to have a good travel plan and I will answer your questions about the traveling spot and search for the best plan about the traveling route and hotel.")]

        print(" ============ Welcome to the TravelAgent Chat! Type 'exit' to stop chatting. ==========")  
        while True:  
            user_input = input(f"User: ")  
            if user_input.lower() == 'exit':  
                print("Goodbye!")  
                break  
            formatted_history = " ".join([f"User: {user}\nSystem: {sys}\n" for user, sys in history])
            response = self.generate_response(formatted_history+"\n"+user_input)  
            print(f"TravelAgent: {response}")  
            print(" ======================================= ")
            
            history.append((user_input, response))
            
            
            
    def stream_chat(
        self, 
        prompt: str, 
        max_length: int = 2048, 
        temperature: float = 0.7, 
        top_p: float = 0.9
        ):  
        """流式聊天功能，逐步返回响应"""  
        # 编码输入  
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)  
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  
        
        generated_ids = inputs['input_ids'].clone()  # 保存完整生成序列  
        attention_mask = inputs['attention_mask'].clone()  
        

        # 生成响应，使用流式生成  
        for _ in range(max_length):
            with torch.no_grad():  
                outputs =  self.model.generate(  
                    input_ids = generated_ids,  
                    attention_mask = attention_mask,
                    max_length=max_length,  
                    temperature=temperature,  
                    top_p=top_p, 
                    max_new_tokens=1,  # 每次生成1个新token
                    do_sample=True,  
                    pad_token_id=self.tokenizer.pad_token_id,  
                    eos_token_id=self.tokenizer.eos_token_id,  
                    # 使用`output_scores=True`和`return_dict_in_generate=True`来启用流式生成  
                    output_scores=True,  
                    return_dict_in_generate=True,  
                    # 启用流式生成  
                    num_return_sequences=1,  
                    # 可以在这里设置其他流式生成的参数  
                    output_hidden_states=False  
                )
                    
                '''
                output.type = GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
                '''
                    
                    
            # 获取新生成的token 
            new_token = outputs.sequences[0, -1].item() 
            # new_word = self.tokenizer.decode(new_token, skip_special_tokens=True) 
            
            if new_token == self.tokenizer.eos_token_id:
                break 
            
            # 更新序列和注意力掩码 
            generated_ids = torch.concat([generated_ids, new_token.unsqueeze(-1)], dim=-1)
            
            attention_mask = torch.concat([
                attention_mask,
                torch.ones((1,1), device = self.device, dtype = attention_mask.dtype)], dim=-1)

            # 解码并返回当前结果
            current_response = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )[len(prompt):]
            
            yield current_response