## 基于huggingface大模型微调+RLHF+RAG的旅游路径规划智能体

## 实验报告
有 实验报告.pdf 可供了解。
请注意：报告中仅描述了该项目的早期版本，很多重要模块都没加

## 环境配置

注册HuggingFace
配置HF_TOKEN到环境变量

```python
conda create -name 'xxx' python=3.8

conda activate xxx

pip install sklearn
pip install torch
pip install numpy
pip install pandas

pip install transformers
pip install langchain
pip install peft
pip install datasets
pip install bitsandbytes
```



## 模型权重烦请手动下载到 Travel-Agent-based-on-LLM-and-SFT/model_weights 目录下

目前的我的主机上是下好了这3个模型
![image](https://github.com/user-attachments/assets/29ae0f5c-c6ff-462b-a4bf-a895e32765bc)

gemma_2b用来微调+推理

其余两个用来做word embedding, 至于用哪个，如果GPU<=A10, 直接用bert-base-chinese







## 主要功能

fine_tune_xxx.py 的文件都是用来微调的，有些是仅微调，有些是量化+微调


rag_xxx.py 的文件都是用来做RAG的




## 如何运行

```shell
python fine_tune_gemma_2b.py
```


```shell
python rag_naive.py
```







## RAG运行结果
![image](https://github.com/user-attachments/assets/ceec5972-c689-47ba-91d9-9df160e54dd8)
![image](https://github.com/user-attachments/assets/27aea7e5-620b-42dd-a68d-070c5c0be2cb)
![image](https://github.com/user-attachments/assets/577a138a-f3e7-48f0-bd97-e319ae7982c7)


#### 运行结果解释
我们给RAG的问题包含了：question+context， context是由数据集中前5个与question最接近的样本组成的。
