from datasets import load_dataset



train_split = load_dataset("csv",data_files="datasets/bitext-travel-llm-chatbot-training-dataset.csv", split="train")

# eval_split = load_dataset("csv",data_files="datasets/bitext-travel-llm-chatbot-training-dataset.csv", split="test")



print(train_split[0])