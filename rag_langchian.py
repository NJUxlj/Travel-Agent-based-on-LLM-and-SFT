import langchain

# from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI


# llm=Ollama(model = 'llama3' )

# llm.invoke('Tell me a joke')



llm = ChatOpenAI()
# llm = ChatOpenAI(api_key="...")
llm.invoke("how can langsmith help with testing?")


