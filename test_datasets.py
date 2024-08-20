from datasets import load_dataset



dataset = load_dataset("csv",data_files="datasets/bitext-travel-llm-chatbot-training-dataset.csv")


# dataset1 = dataset.to_iterable_dataset()

# eval_split = load_dataset("csv",data_files="datasets/bitext-travel-llm-chatbot-training-dataset.csv", split="test")


# train_test_split = dataset.train_test_split(test_size=0.2)

train_split = load_dataset("csv",data_files="datasets/bitext-travel-llm-chatbot-training-dataset.csv", split = "train[-20%:]")
test_split = load_dataset("csv",data_files="datasets/bitext-travel-llm-chatbot-training-dataset.csv",split = "train[:80%]")

print("train_split[0] = \n",train_split[0])

print("=======")


print("test_split[0] = \n",test_split[0])




# 构建10折交叉验证数据集
validation_set = load_dataset("csv",data_files="datasets/bitext-travel-llm-chatbot-training-dataset.csv", 
                                        split =[f"train[{i}%:{i+10}%]" for i in range(0, 100, 10)])

training_set = load_dataset("csv",data_files="datasets/bitext-travel-llm-chatbot-training-dataset.csv", 
                                        split =[f"train[:{i}%]+train[{i+10}%:]" for i in range(0, 100, 10)])

print("=====================================")
print()

print("validation_set[0] = \n",validation_set[0])

print("training_set[0]=\n",training_set[0])




# 手动更改特征的名字：
remove_columns = ['intent','category','tags']
shrinked_dataset =dataset.map(remove_columns= remove_columns)


# 查看训练集的所有特征
print("========================")
print(train_split.features)
print("========================")


from transformers import AutoTokenizer

tokenizer_id = "model_weights/bert-base-chinese"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)  


# def tokenization(examples, field):
#     return tokenizer(examples[feild])


final_dataset = shrinked_dataset.map(lambda examples:tokenizer(examples['instruction']))
final_dataset = final_dataset.map(lambda examples:tokenizer(examples['response']))




print("final_dataset['train'][0] = \n",final_dataset['train'][0])





def remove_mask(examples):
    import re
    examples['response'] = re.sub(r"\{\{.*?\}\}", "",examples['response'])

    print("examples = \n",examples)
    return examples




unmasked_dataset = dataset.map(remove_mask)

print()
print(unmasked_dataset['train'][0])





# Convert to iterable dataset


# iterable_dataset = dataset.to_iterable_dataset(num_shards=64)
# print(iterable_dataset['train'][0])


# iterable_dataset2 = load_dataset("csv",data_files="datasets/bitext-travel-llm-chatbot-training-dataset.csv",streaming=True)
# print(iterable_dataset['train'][1])




new_dataset = dataset.rename_columns({"instruction":"order", "response":"answer"})


# new_dataset = new_dataset.to_iterable_dataset().shuffle(seed=42, buffer_size=1000)

print(new_dataset.features)