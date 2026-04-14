import requests  
import json  
import re  
from typing import Dict, Any, List
import os

from bs4 import BeautifulSoup 
# from baidusearch.baidusearch import search
from urllib.parse import quote_plus  
import time
import random

from datetime import datetime  


from src.tools.prompt_template import MyPromptTemplate



# os.environ['https_proxy'] = 'http://127.0.0.1:7890'
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'



class ToolDispatcher:  
    def __init__(self):  
        self.executors = {  
            # "google_search": GoogleSearchExecutor(  
            #     api_key=CUSTOM_SEARCH_API_KEY,  
            #     search_engine_id=CUSTOM_SEARCH_ENGINE_ID  
            # ),
            
            "get_weather": WeatherExecutor,
            "book_hotel": HotelExecutor,
            "book_flight": PlaneTicketExecutor,
            "find_fastest_route": TransportationExecutor,
            
        }  
        
        self.prompt_template = MyPromptTemplate()
    
    def parse_tool_call(self, tool_str: str) -> Dict:  
        """解析工具调用字符串"""  
        pattern = r"(\w+)\((.*)\)"  
        match = re.match(pattern, tool_str)  
        if not match:  
            return None  
        
        tool_name = match.group(1)  
        args_str = match.group(2)  
        
        # 解析参数键值对  
        args = {}  
        for pair in re.findall(r"(\w+)=([^,]+)", args_str):  
            key = pair[0]  
            value = pair[1].strip("'")
            if re.match(r'^-?\d+$', value):  # 支持负整数
                value = int(value)
            args[key] = value  
        
        return {"tool": tool_name, "args": args}  

    def execute(self, tool_call: str) -> Dict:  
        """执行工具调用"""  
        parsed = self.parse_tool_call(tool_call)  
        if not parsed:  
            return {"error": "Invalid tool format"}  
        
        executor = self.executors.get(parsed["tool"])  
        if not executor:  
            return {"error": "Tool not registered"}  
        
        # 获取工具参数规范  
        tool_template = self.prompt_template.tools.get(parsed["tool"])  
        if not tool_template:  
            return {"error": "Tool template not found"}  
        
        
        # 参数类型校验
        for param in tool_template.parameters:
            if param.required and param.name not in parsed["args"]:
                return {"error": f"Missing required parameter: {param.name}"}
            if param.name in parsed["args"]:
                expected_type = param.type
                actual_value = parsed["args"][param.name]
                # 使用安全的类型检查方式替代eval
                type_map = {
                    "int": int,
                    "float": float,
                    "str": str,
                    "bool": bool,
                    "list": list,
                    "dict": dict,
                }
                expected = type_map.get(expected_type, object)
                if not isinstance(actual_value, expected):
                    return {"error": f"Type mismatch for {param.name}, expected {expected_type}"}  
        
        print( "parse_args = ", parsed["args"])
        
        # parsed["args"] = {"query":..., "max_results":...}
        return executor.execute(**parsed["args"]) 




class GoogleSearchExecutor:  
    def __init__(self, api_key: str, search_engine_id: str):  
        self.base_url = "https://www.googleapis.com/customsearch/v1"  
        self.api_key = api_key  
        self.search_engine_id = search_engine_id  

    def execute(self, query: str, max_results: int = 5) -> Dict[str, Any]:  
        params = {  
            "key": self.api_key,  
            "cx": self.search_engine_id,  
            "q": query,  
            "num": max_results  
        }  
        
        try:  
            response = requests.get(self.base_url, params=params)  
            response.raise_for_status()  
            return self._parse_results(response.json())  
        except Exception as e:  
            return {"error": str(e)}  

    def _parse_results(self, data: Dict) -> Dict:  
        """解析Google API响应"""  
        return {  
            "items": [{  
                "title": item.get("title"),  
                "link": item.get("link"),  
                "snippet": item.get("snippet")  
            } for item in data.get("items", [])]  
        }  


class WeatherExecutor:  
    def __init__(self, api_key: str = ""):  
        self.base_url = "https://api.weatherapi.com/v1/history.json"  
        self.api_key = api_key  # 实际使用时需要API密钥  

    def execute(self, location: str, date: str) -> Dict[str, Any]:  
        """模拟天气数据查询"""  
        try:  
            # 验证日期格式  
            datetime.strptime(date, "%Y-%m-%d")  
            
            # 模拟数据（实际应调用API）  
            return {  
                "location": location,  
                "date": date,  
                "temperature": random.randint(15, 30),  
                "condition": random.choice(["晴", "多云", "小雨"]),  
                "humidity": f"{random.randint(40, 80)}%"  
            }  
        except ValueError:  
            return {"error": "Invalid date format, use YYYY-MM-DD"}  
        except Exception as e:  
            return {"error": str(e)}  

class HotelExecutor:  
    def __init__(self, api_key: str = ""):  
        self.api_key = api_key  # 酒店API密钥  

    def execute(self, location: str, check_in: str, check_out: str,   
               budget: int = None, room_type: str = None) -> Dict[str, Any]:  
        """酒店预订模拟"""  
        try:  
            # 参数验证  
            datetime.strptime(check_in, "%Y-%m-%d")  
            datetime.strptime(check_out, "%Y-%m-%d")  
            
            # 模拟响应（实际应调用API）  
            hotels = [{  
                "name": f"{location}酒店示例{1}",  
                "price": random.randint(300, 800),  
                "room_type": room_type or "大床房",  
                "rating": round(random.uniform(3.5, 5.0), 1)  
            } for _ in range(3)]  
            
            if budget:  
                hotels = [h for h in hotels if h["price"] <= budget]  
                
            return {"available_hotels": hotels}  
        except ValueError:  
            return {"error": "Invalid date format"}  
        except Exception as e:  
            return {"error": str(e)}  

class PlaneTicketExecutor:  
    def __init__(self, api_key: str = ""):  
        self.api_key = api_key  # 航班API密钥  

    def execute(self, departure: str, destination: str, date: str,  
               seat_class: str = "economy") -> Dict[str, Any]:  
        """航班查询模拟"""  
        try:  
            datetime.strptime(date, "%Y-%m-%d")  
            
            # 模拟航班数据  
            flights = [{  
                "flight_no": f"{random.choice(['MU', 'CA', 'CZ'])}{random.randint(1000,9999)}",  
                "departure_time": f"{random.randint(6,22)}:00",  
                "price": random.randint(500, 2000),  
                "seat_class": seat_class  
            } for _ in range(3)]  
            
            return {  
                "departure": departure,  
                "destination": destination,  
                "flights": sorted(flights, key=lambda x: x["price"])  
            }  
        except ValueError:  
            return {"error": "Invalid date format"}  
        except Exception as e:  
            return {"error": str(e)}  

class TransportationExecutor:  
    def __init__(self, api_key: str = ""):  
        self.api_key = api_key  # 地图API密钥  

    def execute(self, start: str, end: str,   
               transport_type: str = "driving") -> Dict[str, Any]:  
        """交通路线规划模拟"""  
        try:  
            # 模拟不同交通方式  
            base_time = random.randint(20, 60)  
            times = {  
                "driving": base_time,  
                "walking": base_time * 4,  
                "transit": base_time * 1.5  
            }  
            
            return {  
                "start": start,  
                "end": end,  
                "duration": f"{times.get(transport_type, base_time)} 分钟",  
                "distance": f"{random.randint(5, 20)} 公里",  
                "route": [  
                    f"从{start}出发",  
                    "沿模拟路线行驶",  
                    f"到达{end}"  
                ]  
            }  
        except Exception as e:  
            return {"error": str(e)}  