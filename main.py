
try:
   from src.models.model import TravelAgent  
   # from src.ui.app import launch_ui  

   from src.finetune.sft_trainer import SFTTrainer

   from src.utils.utils import SFTArguments

   from src.configs.config import MODEL_PATH, DATA_PATH, SFT_MODEL_PATH

   from src.data.data_processor import TravelQAProcessor


   from src.rag.rag import RAG

   from src.rag.rag import CityRAG



   from src.agents.single_bfs_planning_agent import MyAgent
   
   from src.rag.rag_dispatcher import RagDispatcher
   
   
except Exception as e:
   print("导包出现问题，应该是版本问题，但是先不管: ", str(e))
   print("==================================")

import argparse
import asyncio

PLAN_EXAMPLE =  """

### **一、出发前准备**
1. **签证办理**：
   - 确保持有有效的美国签证（B1/B2旅游签证）。
   - 提前预约签证面试，准备好相关材料（护照、照片、银行流水、行程计划等）。

2. **机票预订**：
   - 提前1-2个月预订机票，选择直飞或转机航班。
   - 推荐航空公司：国航（直飞）、达美航空、美国联合航空等。
   - 直飞航班时长约13小时，转机航班可能需要20小时以上。

3. **住宿预订**：
   - 根据预算选择酒店或民宿，推荐住在曼哈顿中城区（交通便利，靠近主要景点）。
   - 提前通过平台（如Booking、Airbnb）预订。

4. **旅行保险**：
   - 购买覆盖医疗、行李丢失、航班延误等的旅行保险。

5. **行李准备**：
   - 根据季节准备衣物（纽约冬季寒冷，夏季炎热）。
   - 带好转换插头（美国电压为110V，插头为A/B型）。

---

### **二、交通安排**
#### **1. 北京到纽约的航班**
   - **直飞航班**：从北京首都国际机场（PEK）或大兴国际机场（PKX）出发，抵达纽约肯尼迪国际机场（JFK）或纽瓦克自由国际机场（EWR）。
   - **转机航班**：可选择在东京、首尔或欧洲城市（如巴黎、伦敦）中转。

#### **2. 纽约市内交通**
   - **机场到市区**：
     - JFK机场：乘坐AirTrain连接地铁（E线或A线），或选择出租车/网约车（约$60-$80）。
     - EWR机场：乘坐AirTrain连接NJ Transit火车，或选择出租车/网约车（约$70-$90）。
   - **市内交通**：
     - 地铁是纽约最便捷的交通工具，建议购买7天无限次地铁卡（约$34）。
     - 可使用Uber/Lyft等网约车服务。

---

### **三、纽约行程规划**
#### **Day 1：抵达纽约，熟悉环境**
   - 抵达后入住酒店，调整时差。
   - 晚上可前往**时代广场（Times Square）**，感受纽约的繁华夜景。

#### **Day 2：曼哈顿经典景点**
   - 上午：参观**自由女神像**（乘坐渡轮前往自由岛）。
   - 下午：游览**华尔街**、**世贸中心一号楼（One World Observatory）**。
   - 晚上：漫步**布鲁克林大桥**，欣赏曼哈顿夜景。

#### **Day 3：文化与艺术之旅**
   - 上午：参观**大都会艺术博物馆（The Met）**。
   - 下午：游览**中央公园（Central Park）**，骑自行车或步行。
   - 晚上：观看百老汇音乐剧（提前购票）。

#### **Day 4：现代纽约体验**
   - 上午：参观**帝国大厦**或**洛克菲勒中心（Top of the Rock）**观景台。
   - 下午：购物（第五大道、SOHO区）。
   - 晚上：品尝纽约特色美食（如披萨、芝士蛋糕）。

#### **Day 5：布鲁克林与周边探索**
   - 上午：前往布鲁克林区，参观**布鲁克林博物馆**或**布鲁克林植物园**。
   - 下午：漫步**威廉斯堡（Williamsburg）**，体验当地文艺氛围。
   - 晚上：返回曼哈顿，享受最后的夜晚。

#### **Day 6：返程或继续深度游**
   - 根据航班时间安排返程，或选择前往周边城市（如华盛顿、波士顿）继续旅行。

---

### **四、预算参考**
1. **机票**：往返约5000-8000元人民币（视季节和航班而定）。
2. **住宿**：中档酒店每晚约1500-2500元人民币。
3. **餐饮**：每日约300-500元人民币。
4. **景点门票**：
   - 自由女神像渡轮：约$24。
   - 大都会艺术博物馆：建议捐赠$25。
   - 帝国大厦观景台：约$44。
5. **交通**：地铁卡$34，机场接驳交通$20-$40。

---

### **五、注意事项**
1. **时差**：纽约比北京时间晚13小时（夏令时晚12小时）。
2. **小费文化**：餐厅、出租车等需支付15%-20%的小费。
3. **安全问题**：注意保管好个人财物，避免深夜前往偏僻区域。

"""


'''

打开控制台运行
python main.py --function rag_dispatcher

python main.py --function rag_dispatcher --rag_type self_rag

'''

def train():

    # 初始化模型  
    agent = TravelAgent()  

    # # 启动UI  
    # launch_ui(agent)


    # agent.chat()



    # args = SFTArguments()  # 使用parse_args获取参数
    trainer = SFTTrainer(travel_agent = agent)

    processor = TravelQAProcessor(agent.tokenizer)

    processor.load_dataset_from_hf(DATA_PATH)

    trainer.max_length = processor.max_length
    print("trainer.max_length = ", trainer.max_length)



    processed_data = processor.prepare_training_features()

    print("mapping over")



    keys = list(processed_data.keys())

    print("keys = ", keys)

    trainer.train(
        train_dataset=processed_data["train"].select(range(50)),
        eval_dataset=processed_data["train"].select(range(50,80))
    )
    
def inference():
    # model = SFTTrainer.load_trained_model(SFT_MODEL_PATH)
    
    agent = TravelAgent(SFT_MODEL_PATH)
    
    # agent.chat()  
    
    agent.stream_chat("I want to travel.")
    
    
def use_rag():
    agent = TravelAgent(SFT_MODEL_PATH)
    rag = RAG(agent = agent)
    
    # results = rag.query_db("train tickets")
    # print(results)
    
    rag.rag_chat()
    
    
    # I want to go to Florida, I am now in the New York. Please help me book a hotel in Floria. I will arrive at 12:36:42 and leave at 22:43:12, my budget is 5000 dollars.


def use_city_rag():
   
   rag = CityRAG()
   rag.query("帮我想一个上海出行的方案")
      
   # except Exception as e:
   #    print("rag 对象构建出现错误：", str(e))
      
      
      
async def use_rag_dispatcher(rag_type:str="self_rag"):
    rag_dispatcher = RAGDispatcher(rag_type=rag_type)
   
   
    answer = await rag_dispatcher.dispatch("帮我规划一个广州三日游的方案")
    
    print("final answer  = ", answer)

def use_rag_web_demo():
    
    from src.ui.rag_web_demo import initialize_rag, create_demo
    rag_system = initialize_rag()  
    demo = create_demo(rag_system)  
    demo.launch(  
        server_name="0.0.0.0",  
        server_port=7860,  
        share=False,  
        favicon_path="./travel_icon.png"  
    )  
    
    
def use_agent():
    agent = MyAgent()
    

    
    result = agent.get_final_plan(PLAN_EXAMPLE)
    
    print(result)
    
    
async def parse_arguments(default_func = "use_agent"):
      parser = argparse.ArgumentParser(description="Travel Agent: Choose the function you wang to display ~")
      parser.add_argument(
         "--function", 
         type=str, 
         default = default_func, 
         help="Choose the function from [train, inference, use_rag, use_agent, use_rag_web_demo, rag_dispatcher]"
         )
      
      parser.add_argument(
         "--rag_type", 
         type=str, 
         default = "self_rag", 
         help="This is useful when --function==rag_dispatcher, Choose the RAG type from [self_rag, rag, mem_walker]"
         )

      # parser.add_argument(
      #    "--model", type=str, default="gpt-4o", help="model used for decoding. Please select from [gpt-4o, gpt-4o-mini]"
      # )
      

      
      args = parser.parse_args()
      
      if args.function == "train":
          train()

      elif args.function == "inference":
          inference()

      elif args.function == "use_rag":
          use_rag()

      elif args.function == "use_agent":
          use_agent()
      elif args.function == "rag_dispatcher":
          await use_rag_dispatcher(args.rag_type)
      elif args.function == "use_rag_web_demo":
          use_rag_web_demo()
          
          
    
if __name__ == "__main__":
    # inference()
    # use_rag()
    # use_rag_web_demo()
   #  use_agent()
   asyncio.run(parse_arguments())
   
   # use_city_rag()