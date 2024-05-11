from langchain_community.llms import Ollama 

from langchain.chains import create_history_aware_retriever

from langchain_community.document_loaders import WebBaseLoader

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

from langchain_community.embeddings import OllamaEmbeddings

# 向量库
from langchain_community.vectorstores import FAISS
# 分词器模型
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.documents import Document

from langchain.chains import create_retrieval_chain

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage 


# 检索工具包装类
from langchain.tools.retriever import create_retriever_tool

from langchain_openai import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain_openai import OpenAIEmbeddings


from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor


import os

# 代理地址
# os.environ["OPENAI_API_BASE"] = "http://{your proxy url}/v1"
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

llm = llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# text = "⽤中⽂告诉我你能做什么?"
# print(llm(text))



# llm = Ollama(model="gemma:2b")
loader = WebBaseLoader("https://discuss.huggingface.co/t/how-should-a-absolute-beginners-start-learning-ml-llm-in-2024/67655")
docs = loader.load()
# 分词器模型
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
# 文本转向量模型
# embeddings = OllamaEmbeddings()

embeddings = OpenAIEmbeddings()

# 将文本片段转向量后传入向量库
vector = FAISS.from_documents(documents, embeddings)

# 模板
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector.as_retriever()
retrieval_chain =  create_retrieval_chain(retriever, document_chain)


# 检索工具包装器
retriever_tool = create_retriever_tool(
    retriever,
    "large_language_model_search",
    "Search for information about how to learn large_language_model. For any questions about large_language_model, you must use this tool!",
)
    
tools = [retriever_tool]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "新手应该如何学习LLM?"})