from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig,AutoModel
from datasets import load_dataset
import gradio as  gr

from typing import List, Optional, Dict, Literal

from transformers import pipeline

import torch




def load_documents(file_path:Optional[str]=None):
    '''
        从本地数据集加载 旅游路径规划的相关文档

    '''

    travel_dataset = load_dataset("csv", data_files ="datasets/bitext-travel-llm-chatbot-training-dataset.csv" )

    
    remove_columns = ['intent','category','tags']
    travel_dataset =travel_dataset.map(remove_columns= remove_columns)

    result = []
    # features = travel_dataset.features.keys()
    for example in travel_dataset['train']:
        # print("example = ", example)
        result.append("instruction: "+example['instruction']+". \n"+"response: "+example["response"]+". ")

    return result


# documents 是一个包含所有文档文本的列表
def encode_documents(documents: List[str], tokenizer,model):
    torch.cuda.empty_cache()
    encoded_docs = []
    index = 0
    for doc in documents:
        doc_sequence = tokenizer(doc, return_tensors="pt")
        doc_embedding = model(**doc_sequence)
        encoded_docs.append(doc_embedding.pooler_output)

        index+=1
        if index>100:
            break
    return encoded_docs




def encode_question(question, tokenizer,model):
    '''
     把query句子转为句向量
    '''
    torch.cuda.empty_cache()
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model(**inputs)
    question_embedding = outputs.pooler_output  # 获取句向量
    return question_embedding


from sklearn.metrics.pairwise import cosine_similarity

def search_documents(question_embedding, document_embeddings:List[torch.Tensor], top_k = 5):
    '''
      在文档中查询与问题匹配的文档句

    '''
    print("=============== question embedding =========================")
    # print(question_embedding)

    print("====================== document embedding ===================")
    # print(document_embeddings)

    # similarities = cosine_similarity(question_embedding,document_embeddings)
    question_embedding = question_embedding.tolist()
    similarities = []

    print(type(question_embedding))
    print(type(document_embeddings))

    for doc_embedding in document_embeddings:
        doc_embedding = doc_embedding.tolist()
        simi = cosine_similarity(question_embedding, doc_embedding)
        similarities.append(simi)


    # print("similarities = ",similarities)

    similarities: List[array]

    similarities = [array[0][0] for array in similarities]


    # print(similarities)


    similarities:torch.FloatTensor = torch.FloatTensor(similarities)

    top_k = min(top_k, len(similarities))
    # [::-1] 反转索引
    top_k_indices = similarities.argsort()[-top_k:][::1]

    # top_k_indices = reversed(top_k_indices)

    print("top_k_indices = ", top_k_indices)

    return top_k_indices



def get_relevant_documents(top_k_indices:torch.FloatTensor, doc_strings:List[str]):
    results = []

    for index in top_k_indices:
        results.append(doc_strings[index])

    print(type(results))

    return results





def generate_answer(question:str, relevant_documents:List[str], tokenizer:AutoTokenizer, generator):

    context = " ".join(relevant_documents)

    input_text = f"question: {question} \n context: {context}"

    # inputs = tokenizer(input_text, return_tensors="pt")
    answer = generator(input_text, max_length=1200, num_beams=4)[0]['generated_text']

    # answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer 





def main(query:Optional[str]=None):
    '''
            把所有模块整合到一起
    '''



    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 用别人定义的word2Vec模型
    model_id = "model_weights/bert-base-chinese"



    tokenizer_id = "model_weights/bert-base-chinese"


    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )



    model  = AutoModel.from_pretrained(model_id)
    # model.to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    print("===============模型加载完毕 ====================")
    print("=================================================")


    doc_strings=load_documents()
    print("================== 文档加载完毕 ================")

    question = "你是谁"
    if query is not None:
        question = query 

    question_embedding = encode_question(question, tokenizer, model)
    doc_embeddings=encode_documents(doc_strings, tokenizer, model)


    top_k_indices = search_documents(question_embedding, doc_embeddings)

    relevant_documents = get_relevant_documents(top_k_indices, doc_strings)

    # 选择预训练的文本生成模型
    generator = pipeline("text2text-generation", model="model_weights/gemma-2b")

    answer = generate_answer(question, relevant_documents, tokenizer, generator)



    print("RAG智能体的输出为： \n",answer)





def launch_on_gradio():
    RAGui = gr.Interface(
        fn=main,
        inputs=gr.Textbox(lines=2, placeholder = "Enter your query here..."),
        outputs = "text",
        title = "RAG Agent",
        description="Ask a question and get an answer based on simple document retrieval."

    )

    RAGui.launch()


if __name__ =='__main__':
    # main()
    launch_on_gradio()

   


