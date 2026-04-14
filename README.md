## 基于Qwen2.5+LoRA微调+RLHF+RAG的旅游路径规划智能体

#### 注：
- 如果各位运行项目出现问题无法解决的话，可以直接给我提issue，我会抽空fix的。
- 最好附上报错截图。

#### 近期修复：

- **2026-04-15** 代码全面审查与修复：共修复26个问题
  - 训练器修复：ppo/dpo/grpo/sft初始化、参数误用、语法错误、混合精度
  - RAG系统：corrective_rag实现、rag()方法、导入路径、数据库初始化逻辑
  - 安全修复：移除tool_executor中的eval安全风险
  - 代码质量：环境变量配置、循环导入、代码重复、梯度检查点
  - 导入路径：修复main.py、rag.py、rag_web_demo.py等处的所有导入错误


### 项目的组成部分
- 本项目希望借助轻量级大模型`Qwen2.5-0.5B~3B`帮助用户在本地更好地规划旅行路径，提高旅行体验。本项目是基于Qwen2.5+SFT微调+RLHF+RAG的旅游路径规划智能体。

- 项目的整体框架如下：
```bash
- Qwen2模型本体

- SFT + RLHF 系统
    - SFTTrainer
    - PPOTrainer
    - DPOTrainer
    - GRPOTrainer
- RAG系统
    - RAG 调度器
        - naive RAG
        - Self-RAG
        - MemWalker-RAG
- multi-agent 系统

```


### RAG系统
---
**辅助工具**：
- 工具调用系统 (Google Search + Weather API + Hotel Booking API + Plane Ticket API + Shortest Path API) 
- 自定义的Prompt模板, 继承了 query+context+文档库匹配段落+工具列表+工具格式。它可以让大模型返回用户query命中的工具的函数API字符串. 
- ToolDispatcher: 工具调度器，它可以解析用户query命中的工具的函数API字符串，随后使用ToolExecutor对象调用对应的工具函数。
- ToolExecutor: 实际调用各种工具API的执行器。
- ChatPDF

传统RAG：
- 基于传统文档匹配+BM25实现的RAG
- 完全基于Langchain实现的RAG


**MemWalker** [一种RAG方法，可以自己去搜paper]:
- 记忆树构建过程
    - 将长文本分割成适合LLM上下文窗口的小块
    - 使用LLM为每个段落生成摘要节点 (summary node / leaf node)
    - 递归地将这些摘要进一步汇总形成高层次摘要节点 (parent node)
    - 最终构建完整的树形记忆结构

- 交互式导航机制
    - 从树的根节点开始导航
    - LLM检查不同文本部分，确定与查询相关的路径, 最后LLM会通过相似度比较，一直走到某个叶节点。
    - 最后通过自我纠错能力，判断该叶节点是不是需要的答案，如果不是，则回溯到父节点，继续搜寻下一个叶节点。

- 在遍历过程中维护工作记忆
    - 能够从早期导航步骤中的错误中恢复, 通过让模型在叶节点输出 action=-1 / -2 / 0 这样的标识，来决定是否回溯。如果回溯，需要去【工作记忆】中找到父节点。
    - 包含对导航决策的推理能力 【自主的路径分支判断】


**Self-RAG**:
-  自适应检索机制
    - 模型可按需检索信息（可多次检索或完全跳过检索）
    - 避免传统RAG中固定数量检索的局限性
    - 动态决定何时需要外部知识支持 [模型通过是否输出一个 [Retrival] token 来决定是否需要检索外部知识源]

- 反思标记系统
    - 使用特殊`反思标记[reflection tokens]`评估生成内容
    - 对检索段落和自身生成内容进行批判性分析
    - 标记内容的相关性、支持度和完整性 (如何标记？使用模型输出的3种反思token：[IS_SUP] [IS_REL] [IS_COM])

- 三步执行流程
    - 步骤1：决定是否需要检索（生成Retrival标记）
    - 步骤2：同时处理多个检索段落，评估相关性并生成响应
    - 步骤3：生成批评标记(Critic Token)，评价输出质量并选择最佳答案

- 端到端训练方法
    - 训练单一语言模型同时处理检索、生成和批评任务
    - 使模型学会何时检索，如何批判自己的输出



**RAG Dispatcher**
- 可以通过用户传入的参数来选择调用 MemWalker, Self-RAG, 或是传统RAG


### 前端UI
 - Gradio




- 数据预处理系统




## 环境配置
**GPU**: RTX3090 x 2
**Platform**: AutoDL
- NAME="Ubuntu"
- VERSION="20.04.4 LTS (Focal Fossa)"
- CUDA=12.4
- Pytorch=2.5.0

```shell
pip install -r requirements.txt
```


## Download SFT dataset
```bash

cd src/data

huggingface-cli download --resume-download JasleenSingh91/travel-QA --local-dir travel-QA


```


## Download DPO dataset
- 找不到旅游相关的DPO数据集，除非自己做。
- 暂时拿这个冒充一下

```bash
cd src/data

huggingface-cli download --resume-download --repo-type dataset HumanLLMs/Human-Like-DPO-Dataset --local-dir Human-Like-DPO-Dataset

```

## Download GRPO dataset
- 和DPO用的是一样的数据集
```bash
cd src/data

huggingface-cli download --resume-download --repo-type dataset HumanLLMs/Human-Like-DPO-Dataset --local-dir Human-Like-DPO-Dataset
```


## Download RAG dataset

```bash
cd src/data

huggingface-cli download --resume-download BruceNju/crosswoz-sft --local-dir crosswoz-sft

```

## Data Example Format
- `JasleenSingh91/travel-QA`
```Plain Text

{"question":.......,"response":.....}


```




- `crosswoz`
- crosswoz用在哪里：在RAG系统统中，我们会用query去匹配crosswoz中的history这个字段下的数据，作为对已有的context和tool-use response 的补充。
```Plain Text
{
            "dialog_id": "391",
            "turn_id": 0,
            "role": "usr",
            "content": "你好，麻烦帮我推荐一个门票免费的景点。",
            "dialog_act": "[[\"General\", \"greet\", \"none\", \"none\"], [\"Inform\", \"景点\", \"门票\", \"免费\"], [\"Request\", \"景点\", \"名称\", \"\"]]",
            "history": "[]",
            "user_state": "[[1, \"景点\", \"门票\", \"免费\", true], [1, \"景点\", \"评分\", \"5分\", false], [1, \"景点\", \"地址\", \"\", false], [1, \"景点\", \"游玩时间\", \"\", false], [1, \"景点\", \"名称\", \"\", true], [2, \"餐馆\", \"名称\", \"拿渡麻辣香锅(万达广场店)\", false], [2, \"餐馆\", \"评分\", \"\", false], [2, \"餐馆\", \"营业时间\", \"\", false], [3, \"酒店\", \"价格\", \"400-500元\", false], [3, \"酒店\", \"评分\", \"4.5分以上\", false], [3, \"酒店\", \"周边景点\", [], false], [3, \"酒店\", \"名称\", \"\", false]]",
            "goal": "[[1, \"景点\", \"门票\", \"免费\", false], [1, \"景点\", \"评分\", \"5分\", false], [1, \"景点\", \"地址\", \"\", false], [1, \"景点\", \"游玩时间\", \"\", false], [1, \"景点\", \"名称\", \"\", false], [2, \"餐馆\", \"名称\", \"拿渡麻辣香锅(万达广场店)\", false], [2, \"餐馆\", \"评分\", \"\", false], [2, \"餐馆\", \"营业时间\", \"\", false], [3, \"酒店\", \"价格\", \"400-500元\", false], [3, \"酒店\", \"评分\", \"4.5分以上\", false], [3, \"酒店\", \"周边景点\", [], false], [3, \"酒店\", \"名称\", \"\", false]]",
            "sys_usr": "[43, 18]",
            "sys_state": "",
            "sys_state_init": ""
      }
```

## 如何运行

- RAG

```shell
# 使用最基础的 RAG
python main.py --function use_rag

# 使用调度器进行RAG类型选择
python main.py --function rag_dispatcher --rag_type self_rag
```

- 测试SFT训练
```bash
python main.py --function train
```

- 单独测试PPO-from-scratch
    - 注意，运行这个脚本之前请自行下载 [reward model权重](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2)，并在 `config.py` 中配好 `REWARD_MODEL_PATH`

```bash
cd /root/autodl-tmp/models

huggingface-cli download --resume-download OpenAssistant/reward-model-deberta-v3-large-v2 --local-dir reward-model-deberta-v3-large-v2
```

```bash
python -m src.finetune.ppo_train_from_scratch
```

- 单独测试PPO训练
    - 同样的，先去下好 reward model 的权重 
```bash
python -m src.finetune.ppo_trainer

```

- 单独测试DPO训练
```bash
python -m src.finetune.dpo_trainer
```

- 单独测试GRPO训练
```bash
python -m src.finetune.grpo_trainer
```



## Experiment Setup
### Model
1. 我们使用了Qwen2.5作为LLM模型
    - 目前仅测试了Qwen2.5的1.5B参数版本

### Project Structure
1. 核心代码都放在 **`src/`** 目录下.
2. **`src/`** 的目录结构：

```yaml
src:
    data:
     - processed_data
     - data_augmentation.py
     - data_preprocessor.py
     - crosswoz-sft:
        - ...
    - travel_qa:
        - ...
     - init.py
    configs:
     - config.py
     - ds_config.py
    agents:
     - travel_knowledge:
        - tour_pages
        - tour_pdfs
     - agent.py
     - bm25.py
     - tools.py # 各种 Executors: google,weather, transportation; 以及 ToolDispatcher
     - zhipuAPI.py
     - prompt_template.py
     - chat_pdf.py 
     - rag.py
     - mem_walker.py
     - self_rag.py
     - corrective_rag.py # 现在还没写好这个，别跑！
     - rag_dispatcher.py
    finetune:
     - dpo_trainer.py
     - sft_trainer.py  # SFTTrainer
     - ppo_trainer.py
     - grpo_trainer.py
     - multi_task_trainer.py
     - __init__.py
    pretrain:
     - pretrain.py
     - init.py
    models:
     - qwen2
        - modeling_qwen2.py
        - configuration_qwen2.py
        - tokenization_qwen2.py
        - dola_decode.py
     - model.py # TravelAgent类所在的位置
     - init.py
    ui:
     - app.py
     - mindmap.py
     - init.py

data:
     - 各种数据集
utils.py
configs:
     - config.py
     - init.py
```


## python script function explanation
1. **`src/agents/agent.py`**





# Results
---
- 如果下面的图里没有，请自行去 `image` 文件夹中找你想要的结果

## SFT Running Snapshot
![SFT](image/SFT_train_snapshot.png)



## 根据 Travel Agent 的路线规划生成思维导图（Mind Map）运行结果
![image](image/mindmap_output.png)





## RAG运行结果

|  |  |  |
|:---:|:---:|:---:|
| ![part1](image/regular_rag/rag1.png) | ![part2](image/regular_rag/rag3.png) | ![part3](image/regular_rag/rag4.png) |
| 图1 | 图2 | 图3 |


- 如你所见，还有一些bug。比如工具调用无法从response中解析出来，有空我来解决。



## Self-RAG 运行结果
![image](image/self_rag/self_rag1.png)


## Mem-Walker-RAG 运行结果

- 运行结果 part1
![image](image/mem_walker/memwalker1.png)


- 运行结果 part2
![image](image/mem_walker/memwalker2.png)


- 运行结果 part3
![image](image/mem_walker/memwalker_final_output.png)


## RAG Web Demo 运行结果
![rag_web_demo](image/rag_web_demo.png)
![rag_web_demo2](image/rag_web_demo2.png)




## PPO from scratch 运行结果
![ppo_from_scratch](image/ppo/ppo_from_scratch.png)



## PPOTrainer训练过程
![ppo_from_scratch](image/ppo/ppo_trainer_running.png)



## DPOTrainer训练过程
![image](image/dpo/dpo_trainer_running.png)



## SFT Evaluation Result
```Plain Text
Epoch  Rouge1   Rouge2  RougeL  BLEU


```


## Citation
- we refer to many other projects when building this project.
- [knowledge-graph-from-GPT](https://github.com/tomhartke/knowledge-graph-from-GPT.git)
- [ai-travel-agent](https://github.com/nirbar1985/ai-travel-agent.git)
- [GPT2](https://github.com/affjljoo3581/GPT2.git)
- [RLHF_instructGPT](https://github.com/LanXiu0523/RLHF_instructGPT.git)
- [PPO_reproduce]()
- [GraphRAG_Paper](https://arxiv.org/pdf/2404.16130)
- [GraphRAG_Repo](https://github.com/microsoft/graphrag.git)

- Dataset Citation:

```bibtext
@inproceedings{zhu2020crosswoz,  
    title={CrossWOZ: A Large-Scale Chinese Cross-Domain Task-Oriented Dialogue Dataset},  
    author={Zhu, Qi and Zhang, Zheng and Fang, Yan and Li, Xiang and Takanobu, Ryuichi and Li, Jinchao and Peng, Baolin and Gao, Jianfeng and Zhu, Xiaoyan and Huang, Minlie},  
    booktitle={Transactions of the Association for Computational Linguistics},  
    year={2020},  
    url={https://arxiv.org/abs/2002.11893}  
}
```

## Cite this Repo
- if you want to use this work in your own project, please cite:

```bibtex
@misc{travel_agent_qwen2_rlhf,  
  author       = {NJUxlj},  
  title        = {Travel-Agent-based-on-Qwen2-RLHF},  
  year         = {2025},  
  howpublished = {\url{https://github.com/NJUxlj/Travel-Agent-based-on-Qwen2-RLHF.git}},  
  note         = {A travel agent fine-tuned using SFT, DPO, PPO, and GRPO with a traveling Q\&A dataset. Includes mindmap generation and LangChain integration.}  
}  
```