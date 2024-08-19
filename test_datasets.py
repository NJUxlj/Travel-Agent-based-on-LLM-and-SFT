from datasets import load_dataset



dataset = load_dataset("csv",data_files="datasets/bitext-travel-llm-chatbot-training-dataset.csv")

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
#dataset = load_dataset('csv', data_files=file_dict, delimiter=';', column_names=['text', 'label'], features=emotion_features)



# 查看训练集的所有特征
print("========================")
print(train_split.features)
print("========================")
