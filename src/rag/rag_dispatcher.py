from pathlib import Path
import os, sys
sys.path.append(Path(__file__).parent)
from mem_walker import MemoryTreeNode
from mem_walker import MemoryTreeBuilder
from mem_walker import ChatPDFForMemWalker
from mem_walker import Navigator
from self_rag import SelfRAG
from rag_config import RAGType
from typing import Literal, Callable, Dict, Tuple

from rag.rag import RAG
from models.model import TravelAgent
from configs.config import PDF_FOLDER_PATH, RAG_DATA_PATH, EMBEDDING_MODEL_PATH
from zhipuai import ZhipuAI

import asyncio

#  input: 旅游规划路径


class RAGDispatcher():
    def __init__(self, rag_type:Literal["rag","self_rag", "corrective_rag", "mem_walker"]="mem_walker"):
        self.rag_type = rag_type

    async def dispatch(self, query:str):
        # 1. 规划路径分析
        # 2. 规划路径执行
        # 3. 规划路径总结
        
        if self.rag_type == RAGType.MEM_WALKER:
            return await self.mem_walker(query)
        
        elif self.rag_type == RAGType.RAG:
            return self.rag(query)

        elif self.rag_type == RAGType.SELF_RAG:
            return await self.self_rag(query)
        elif self.rag_type == RAGType.CORRECTIVE_RAG:
            return self.corrective_rag(query)
        
    def rag(self, query:str):
        """Standard RAG: retrieve relevant documents from database and return them."""
        agent = TravelAgent()
        rag = RAG(agent=agent, use_db=True, use_api=True)

        results = rag.query_db(query, n_results=5)
        return results
    
    async def mem_walker(self,query:str)->str:
        builder = MemoryTreeBuilder()
        
        pdf_reader = ChatPDFForMemWalker()
        pdf_reader.ingest_all(pdf_folder_path=PDF_FOLDER_PATH)
        
        all_chunks = pdf_reader.get_memwalker_chunks()
        root = await builder.build_tree(all_chunks, model_type="api")
        
        builder.print_memory_tree(root)
    
        navigator = Navigator(model_type="api")
        answer = await navigator.navigate(
            root, 
            query
            )
        
        
        return answer
        
        
    
    
    
    async def self_rag(self, query:str):
        rag = SelfRAG(model_type="api")  
        chain = await rag.build_chain()  
        
        result = await chain.ainvoke(query)  
        print(f"最终答案：{result}")  
    
    
    
    def corrective_rag(self, query:str):
        """Corrective RAG: query with original query, if results are poor, reformulate and retry."""
        # Initialize RAG components for database querying
        agent = TravelAgent()
        rag = RAG(agent=agent, use_db=True, use_api=True)

        # First attempt with original query
        initial_results = rag.query_db(query, n_results=5)

        # Check if results meet quality threshold
        if len(initial_results) >= 3 and self._check_quality(initial_results):
            return initial_results

        # If results are insufficient, reformulate the query and retry
        reformulated_query = self._reformulate_query(query, initial_results)
        corrected_results = rag.query_db(reformulated_query, n_results=5)

        # Combine and deduplicate results
        combined = self._merge_results(initial_results, corrected_results)
        return combined[:5]

    def _check_quality(self, results: list, min_length: int = 50) -> bool:
        """Check if results meet minimum quality threshold."""
        if not results:
            return False
        avg_length = sum(len(str(r)) for r in results) / len(results)
        return avg_length >= min_length

    def _reformulate_query(self, original_query: str, previous_results: list) -> str:
        """Reformulate query based on previous results to improve retrieval."""
        # Use LLM to reformulate query if available, otherwise use keyword expansion
        context = "\n".join(str(r)[:200] for r in previous_results[:2])

        reformulation_prompt = f"基于以下上下文，优化搜索查询以获得更好的旅行规划结果。\n\n原始查询：{original_query}\n\n相关上下文：{context}\n\n请提供一个更精确的搜索查询（只返回查询语句，不要其他内容）："

        try:
            client = ZhipuAI(api_key=os.environ.get("ZHIPU_API_KEY"))
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": reformulation_prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception:
            # Fallback: append common travel-related terms
            return f"{original_query} 旅游攻略 景点推荐"

    def _merge_results(self, results1: list, results2: list) -> list:
        """Merge and deduplicate results from multiple queries."""
        seen = set()
        merged = []
        for r in results1 + results2:
            key = str(r)[:100]
            if key not in seen:
                seen.add(key)
                merged.append(r)
        return merged

