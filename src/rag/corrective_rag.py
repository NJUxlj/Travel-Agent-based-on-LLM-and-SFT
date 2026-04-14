from typing import List, Tuple, Dict
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from zhipuai import ZhipuAI
import os
import requests  


'''
CorrectiveRAG 还没写完，跑不起来。
'''

class CorrectiveRAG:  
    def __init__(self,   
                 corpus: List[str] = None,
                 corpus_folder_path = None,
                 model_type: str = "api",  
                 correct_threshold: float = 0.59,  
                 incorrect_threshold: float = -0.99):  
        
        assert corpus is not None or corpus_folder_path is not None, "Either corpus or corpus_folder_path must be provided."

        
        if corpus==None:
            self.corpus = self.load_corpus(corpus_folder_path)
        else:
            self.corpus = corpus
        
        self.client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))  
        self.retriever = BM25Retriever.from_texts(self.corpus)  
        
        # 递归字符文本分割器，用于将长文档分割成较小的块。
        # 在knowledge_refinement方法中，它用于将检索到的文档分割成更小的片段，以便进行更精细的相关性评估。
        self.text_splitter = RecursiveCharacterTextSplitter(  
            chunk_size=200,   # 每个文本块的最大长度为200个字符
            chunk_overlap=50   # 相邻文本块之间有50个字符的重叠
        )  
        self.model_type = model_type  
        self.thresholds = {  
            'correct': correct_threshold,  
            'incorrect': incorrect_threshold  
        }  
        
    
    
    def load_corpus(self, corpus_folder_path):
        """Load corpus from a folder containing text files (one per city)."""
        corpus = []
        folder = Path(corpus_folder_path)
        if not folder.exists():
            print(f"Warning: Corpus folder {corpus_folder_path} does not exist.")
            return corpus

        for file_path in folder.glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    corpus.append(content)
        return corpus
    
    
    

    def evaluate_relevance(self, query: str, document: str) -> float:  
        """使用智谱API评估查询与文档相关性"""  
        response = self.client.chat.completions.create(  
            model="glm-4",  
            messages=[{  
                "role": "user",  
                "content": f"请评估以下查询与文档的相关性，返回0-1之间的分数:\n查询：{query}\n文档：{document}"  
            }]  
        )  
        try:  
            return float(response.choices[0].message.content.strip())  
        except:  
            print("\n相关性分数为0，请检查以下内容：")
            print(response.choices[0].message.content)
            print()
            return 0.0  

    def retrieve_documents(self, query: str, k: int = 5) -> List[Document]:  
        """检索相关文档"""  
        return self.retriever.invoke(query)[:k]  

    def knowledge_refinement(self, query: str, documents: List[Document]) -> str:  
        """知识精炼处理"""  
        refined = []  
        for doc in documents:  
            chunks = self.text_splitter.split_text(doc.page_content)  
            for chunk in chunks:  
                score = self.evaluate_relevance(query, chunk)  
                if score > 0.5:  # 过滤阈值  
                    refined.append(chunk)  
        return "\n\n".join(refined[:3])  # 保留前3个相关片段  

    def web_search(self, query: str) -> str:  
        """网络搜索增强"""  
        # 查询改写  
        rewrite_prompt = f"将以下问题转换为搜索关键词（3-5个关键词，用分号分隔）：{query}"  
        keywords = self.client.chat.completions.create(  
            model="glm-4",  
            messages=[{"role": "user", "content": rewrite_prompt}]  
        ).choices[0].message.content.split(";")  
        
        # 执行搜索（示例使用Serper API）  
        headers = {  
            'X-API-KEY': os.getenv("SERPER_API_KEY"),  
            'Content-Type': 'application/json'  
        }  
        response = requests.post(  
            'https://google.serper.dev/search',  
            headers=headers,  
            json={'q': " ".join(keywords)}  
        )  
        return "\n".join([result.get("snippet", "") for result in response.json().get("organic", [])[:3]])  

    def determine_action(self, scores: List[float]) -> str:  
        """确定执行动作"""  
        max_score = max(scores)  
        min_score = min(scores)  
        
        if max_score > self.thresholds['correct']:  
            return "correct"  
        elif min_score < self.thresholds['incorrect']:  
            return "incorrect"  
        else:  
            return "ambiguous"  

    def generate_response(self, query: str, context: str) -> str:  
        """生成最终响应"""  
        response = self.client.chat.completions.create(  
            model="glm-4",  
            messages=[{  
                "role": "user",  
                "content": f"基于以下上下文回答问题：\n{context}\n\n问题：{query}"  
            }]  
        )  
        return response.choices[0].message.content  

    def run(self, query: str) -> str:  
        # 检索阶段  
        documents = self.retrieve_documents(query)  
        
        # 评估阶段  
        scores = [self.evaluate_relevance(query, doc.page_content) for doc in documents]  
        action = self.determine_action(scores)  
        
        # 知识处理  
        if action == "correct":  
            knowledge = self.knowledge_refinement(query, documents)  
        elif action == "incorrect":  
            knowledge = self.web_search(query)  
        else:  
            refined = self.knowledge_refinement(query, documents)  
            web_result = self.web_search(query)  
            knowledge = f"{refined}\n\n{web_result}"  
        
        # 生成阶段  
        return self.generate_response(query, knowledge)  

 
if __name__ == "__main__":  
    corpus = [  
        "蝙蝠侠（1989年电影）的编剧是Sam Hamm...",  
        "亨利·费尔登是英国保守党政治家...",  
        # 添加更多文档...  
    ]  
    
    
    document_path = "src\\agents\\travel_knowledge\\tour_pages"
    crag = CorrectiveRAG(corpus_folder_path=document_path)  
    
    query = "请帮我规划一个杭州一日游路线"  
    response = crag.run(query)  
    print(f"问题：{query}\n回答：{response}")  