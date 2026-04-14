import gradio as gr  
from pathlib import Path
import os, sys

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.rag import RAG  
from src.tools.tool_executor import ToolDispatcher  
from src.models.model import TravelAgent  
from src.configs.config import RAG_DATA_PATH, SFT_MODEL_PATH  

import pandas as pd  
import matplotlib.pyplot as plt  
from typing import Dict 

# from src.finetune.sft_trainer import SFTTrainer



class TrainingMonitor:  
    """实时训练监控器"""  
    def __init__(self):  
        self.loss_history = []  
        self.metric_history = []  
        self.current_step = 0  

    def update(self, logs: Dict):  
        if "loss" in logs:  
            self.loss_history.append((self.current_step, logs["loss"]))  
        if "metrics" in logs:  
            self.metric_history.append((self.current_step, logs["metrics"]))  
        self.current_step += 1  

    def get_loss_plot(self):  
        df = pd.DataFrame(self.loss_history, columns=["step", "loss"])  
        return gr.LinePlot(  
            df,  
            x="step",  
            y="loss",  
            title="训练损失曲线",  
            width=400,  
            height=300  
        )  

    def get_latest_metrics(self):  
        if not self.metric_history:  
            return "等待首次评估..."  
        return pd.DataFrame(  
            self.metric_history[-1][1],  
            index=["最新指标"]  
        )  



# 初始化RAG系统  
def initialize_rag():  
    agent = TravelAgent(model_name=SFT_MODEL_PATH, use_api=True)  
    rag = RAG(  
        agent=agent,  
        dataset_name_or_path=RAG_DATA_PATH,  
        use_db=True,  
        use_prompt_template=True  
    )  
    return rag  


# 创建Gradio界面组件  
def create_demo(rag:RAG):  
    
    
    monitor = TrainingMonitor() 
    
    with gr.Blocks(title="Travel RAG Assistant", theme=gr.themes.Soft()) as demo:  
        gr.Markdown("# 🌍 智能旅行规划助手") 
        
        
        # 聊天界面  
        with gr.Row():  
            with gr.Column(scale=2):  
                chatbot = gr.Chatbot(height=450, label="对话记录")  
                query_box = gr.Textbox(  
                    placeholder="输入您的旅行问题...",  
                    label="用户输入",  
                    lines=3  
                )   
                
                # 示例问题  
                examples = gr.Examples(  
                    examples=[  
                        ["帮我规划上海3日游，预算5000元"],  
                        ["查找北京到巴黎下周最便宜的机票"],  
                        ["推荐杭州西湖附近的4星级酒店"],  
                        ["查询东京下周的天气预报"]  
                    ],  
                    inputs=query_box,  
                    label="示例问题"  
                )  
                
            # 结果显示区域  
            with gr.Column(scale=1):  
                with gr.Tab("工具调用结果"):  
                    tool_output = gr.JSON(label="工具执行详情")  
                with gr.Tab("数据库匹配结果"):  
                    db_output = gr.DataFrame(  
                        headers=["相关结果"],  
                        datatype=["str"],  
                        # max_rows=5,  
                        # overflow_row_behaviour="show_ends"  
                    )  
                with gr.Tab("原始响应"):  
                    raw_output = gr.Textbox(  
                        lines=8,  
                        max_lines=12,  
                        label="完整响应"  
                    )  
            
            # 新增微调控制面板  
            with gr.Column(scale=1):  
                with gr.Tab("模型微调"):  
                    with gr.Accordion("训练参数配置", open=True):  
                        learning_rate = gr.Slider(  
                            minimum=1e-6,  
                            maximum=1e-3,  
                            value=2e-4,  
                            step=1e-6,  
                            label="学习率"  
                        )  
                        num_epochs = gr.Slider(  
                            minimum=1,  
                            maximum=10,  
                            value=3,  
                            step=1,  
                            label="训练轮数"  
                        )  
                        batch_size = gr.Slider(  
                            minimum=1,  
                            maximum=32,  
                            value=4,  
                            step=1,  
                            label="批大小"  
                        )  

            
            
                    
        # 控制按钮  
        with gr.Row():  
            submit_btn = gr.Button("提交", variant="primary")
            clear_btn = gr.Button("清空对话")
        
        with gr.Row():
            with gr.Accordion("语言设置", open=False):  
                lang = gr.Dropdown(  
                    choices=["中文", "English", "日本語"],  
                    value="中文",  
                    label="界面语言"  
                )  
            
        # 处理逻辑  
        def respond(query, chat_history):  
            # 生成响应  
            prompt = rag.prompt_template.generate_prompt(  
                query,   
                "\n".join([f"User:{u}\nSystem:{s}" for u, s in chat_history])  
            )  
            
            # 获取模型响应  
            tool_call_str = rag.agent.generate_response(prompt)  
            
            # 执行工具调用  
            tool_result = rag.dispatcher.execute(tool_call_str)  
            
            # 数据库查询  
            db_result = rag.query_db(query)  
            
            # 生成自然语言响应  
            response = f"{tool_call_str}\n\n工具结果：{tool_result}\n数据库匹配：{db_result[:2]}"  
            
            # 获取最终旅行计划
            travel_plan = rag.get_travel_plan(response)
            
            # 更新对话历史  
            chat_history.append((query, travel_plan))  
            
            return {  
                chatbot: gr.update(value=chat_history),  
                tool_output: tool_result,  
                db_output: [[res] for res in db_result],  
                raw_output: response,  
                query_box: ""  
            }  
            
        # 绑定事件  
        submit_btn.click(  
            fn=respond,  
            inputs=[query_box, chatbot],  
            outputs=[chatbot, tool_output, db_output, raw_output, query_box]  
        )  
        
        clear_btn.click(  
            fn=lambda: ([], [], [], ""),  
            outputs=[chatbot, tool_output, db_output, query_box]  
        )  

    return demo 




if __name__ == "__main__":  
    rag_system = initialize_rag()  
    demo = create_demo(rag_system)  
    demo.launch(  
        server_name="0.0.0.0",  
        server_port=7860,  
        share=False,  
        favicon_path="./travel_icon.png"  
    )  