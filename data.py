from functools import partial

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# Make MyDataset
class MyDataset(Dataset):
    def __init__(self, sentences, labels, method_name, model_name):
        self.sentences = sentences
        self.labels = labels
        self.method_name = method_name
        self.model_name = model_name
        dataset = list()
        index = 0
        for data in sentences:
            tokens = data.split(' ')
            labels_id = labels[index]
            index += 1
            dataset.append((tokens, labels_id))
        self._dataset = dataset

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self.sentences)


# Make tokens for every batch
def my_collate(batch, tokenizer):
    tokens, label_ids = map(list, zip(*batch))

    text_ids = tokenizer(tokens,
                         padding=True,
                         truncation=True,
                         max_length=320,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    # print(1,text_ids['position_ids'])
    # print(2,text_ids['attention_mask'])
    # print(3,text_ids['input_ids'])
    return text_ids, torch.tensor(label_ids)


# 加载数据集
def load_dataset(tokenizer, train_batch_size, val_batch_size, test_batch_size, model_name, method_name, workers):
    # 读取文件
    data = pd.read_csv('simplifyweibo_4_moods.csv', sep=None, header=0, encoding='utf-8', engine='python')
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
