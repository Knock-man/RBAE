from functools import partial

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# 制作数据集
# 形式[([哭,嘴],1),([打，球],0),([看,电,视],1)]   支持索引取样本
class MyDataset(Dataset):
    def __init__(self, sentences, labels, method_name, model_name):
        self.sentences = sentences
        self.labels = labels
        self.method_name = method_name
        self.model_name = model_name
        dataset = list() #
        index = 0 #记录当前样本索引(第几句话)
        for data in sentences:
            tokens = data.split(' ')#将一句话拆分成单词列表 [The,Hanson,brothers,...]
            labels_id = labels[index]#当前索引对应的标签
            index += 1
            dataset.append((tokens, labels_id))#将分割后的单词列表和标签作为元组添加到 dataset 列表中。[([The,Hanson,brothers,...],1)]
        self._dataset = dataset

    def __getitem__(self, index):#支持通过索引获取样本
        return self._dataset[index]

    def __len__(self):#获取样本总数
        return len(self.sentences)


# 每一个batch的数据进行处理
#自定义批处理函数
"""
batch = [
    ([you,will], 1),
    ([you,do,realize], 0),
    ([There,is], 1),
]
"""
def my_collate(batch, tokenizer):
    #print("batch:",batch)
    #batch：这是一个列表，包含从 DataLoader 获取的多个样本。每个样本是一个元组，包含文本数据和标签。
    #tokenizer：这是一个分词器对象，用于将文本数据转换为模型可以理解的格式。
    tokens, label_ids = map(list, zip(*batch)) #map<>
    print("tokens:",tokens)
    print("label_ids:",label_ids)
    """
    tokens = [[you,will], [you,do,realize], [There,is]]
    label_ids = [1, 0, 1]
    """
    #将文本转换为模型可以理解的tokens
    text_ids = tokenizer(tokens,
                         padding=True,#填充，保证序列长度相同
                         truncation=True,#如果长度超过320，截断
                         max_length=320,#序列最大长度
                         is_split_into_words=True,#输入文本已经被分割成单词
                         add_special_tokens=True,#添加特殊的标记，如 [CLS] 和 [SEP]
                         return_tensors='pt')#返回 PyTorch 张量
    print("text_ids",text_ids)
    # print(1,text_ids['position_ids'])
    # print(2,text_ids['attention_mask'])
    # print(3,text_ids['input_ids'])
    return text_ids, torch.tensor(label_ids)


# 加载数据集
def load_dataset(tokenizer, train_batch_size, val_batch_size, test_batch_size, model_name, method_name, workers):
    # 读取文件
    data = pd.read_csv(r'C:\Users\xbj\Desktop\论文阅读\RoBERTa-BiLSTM A Context-Aware Hybrid\参考代码\sentiment_analysis_Imdb\weibo_senti_100k.csv', sep=None, header=0, encoding='utf-8', engine='python')
    len1 = int(len(list(data['labels'])))  # 百分之1的样本长度
    labels = list(data['labels'])[0:len1]  # 同样多的标签
    sentences = list(data['sentences'])[0:len1]  # 百分之1的样本

    # 分割训练集和临时测试集（包含验证集和测试集）
    tr_sen, temp_sen, tr_lab, temp_lab = train_test_split(sentences, labels, train_size=0.6, random_state=42)
    
    # 将临时测试集进一步分为验证集和测试集
    val_sen, te_sen, val_lab, te_lab = train_test_split(temp_sen, temp_lab, test_size=0.5, random_state=42)
    
    # 制作成数据集
    train_set = MyDataset(tr_sen, tr_lab, method_name, model_name)
    val_set = MyDataset(val_sen, val_lab, method_name, model_name)
    test_set = MyDataset(te_sen, te_lab, method_name, model_name)
    
    # DataLoader 批量加载 打乱顺序 多进程多线程加载 自定义数据预处理逻辑
    collate_fn = partial(my_collate, tokenizer=tokenizer)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=workers,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=workers,
                           collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=workers,
                             collate_fn=collate_fn, pin_memory=True)
    
    return train_loader, val_loader, test_loader
