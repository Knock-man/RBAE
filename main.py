import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import logging, AutoTokenizer, AutoModel

from config import get_config
from data import load_dataset
from model import Transformer, Gru_Model, BiLstm_Model, Lstm_Model, Rnn_Model, TextCNN_Model, Transformer_CNN_RNN, \
    Transformer_Attention, Transformer_CNN_RNN_Attention


class Niubility:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info('> creating model {}'.format(args.model_name))
        # 创建模型
        if args.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')#预训练BERT分词器
            self.input_size = 768 #隐藏层大小
            base_model = AutoModel.from_pretrained('bert-base-chinese')#加载了与分词器相匹配的预训练 BERT 模型
        elif args.model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)#add_prefix_space=True 参数是必要的，因为 RoBERTa 的分词器需要在每个 token 前面添加一个空格来正确处理空格。
            self.input_size = 768#隐藏层大小
            base_model = AutoModel.from_pretrained('roberta-base')#加载了与分词器相匹配的预训练 RoBERT 模型
        else:
            raise ValueError('unknown model')
        # Operate the method  下游模型
        if args.method_name == 'fnn':
            self.Mymodel = Transformer(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'gru':
            self.Mymodel = Gru_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'lstm':
            self.Mymodel = Lstm_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'bilstm':
            self.Mymodel = BiLstm_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'rnn':
            self.Mymodel = Rnn_Model(base_model, args.num_classes, self.input_size)
        elif args.method_name == 'textcnn':
            self.Mymodel = TextCNN_Model(base_model, args.num_classes)
        elif args.method_name == 'attention':
            self.Mymodel = Transformer_Attention(base_model, args.num_classes)
        elif args.method_name == 'lstm+textcnn':
            self.Mymodel = Transformer_CNN_RNN(base_model, args.num_classes)
        elif args.method_name == 'lstm_textcnn_attention':
            self.Mymodel = Transformer_CNN_RNN_Attention(base_model, args.num_classes)
        else:
            raise ValueError('unknown method')

        self.Mymodel.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    #打印参数
    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    #训练
    def _train(self, dataloader, criterion, optimizer):#数据迭代器，损失函数，优化器
        train_loss, n_correct, n_train = 0, 0, 0 #累计总训练损失 正确预测数量 样本总数量

        # Turn on the train mode
        self.Mymodel.train()#设置训练模式
        for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii='>='):#进度条
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}#将输入迁移到指定设备(GPU)
            targets = targets.to(self.args.device)#将目标迁移到指定设备
            predicts = self.Mymodel(inputs)# 模型的预测
            loss = criterion(predicts, targets) #模型的损失

            #反向传播和优化
            optimizer.zero_grad()#清除旧梯度
            loss.backward()#计算新梯度
            optimizer.step()#更新模型的权重

            #累积统计信息
            train_loss += loss.item() * targets.size(0) #累计总损失
            n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item() #统计正确预测的数量
            n_train += targets.size(0) #累积训练样本的总数量
        return train_loss / n_train, n_correct / n_train #返回平均损失和准确率
    
    def _val(self, dataloader, criterion):#数据迭代器，损失函数
        val_loss, n_correct, n_val = 0, 0, 0 #累计总测试损失 正确预测数量 总测试样本数量
        # Turn on the eval mode
        self.Mymodel.eval()#开启评估模式
        with torch.no_grad():#禁用梯度计算
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):#进度条
                #数据迁移到设备
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                #模型的预测和损失
                predicts = self.Mymodel(inputs)
                loss = criterion(predicts, targets)

                val_loss += loss.item() * targets.size(0)#累计总损失
                n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()#正确预测数量
                n_val += targets.size(0)#测试样总数

        return val_loss / n_val, n_correct / n_val #返回平均损失和准确率

    #测试函数
    def _test(self, dataloader, criterion):#数据迭代器，损失函数
        test_loss, n_correct, n_test = 0, 0, 0 #累计总测试损失 正确预测数量 总测试样本数量
        
        # Confusion matrix
        TP, TN, FP, FN = 0, 0, 0, 0

        # Turn on the eval mode
        self.Mymodel.eval()#开启评估模式
        with torch.no_grad():#禁用梯度计算
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):#进度条
                #数据迁移到设备
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                #模型的预测和损失
                predicts = self.Mymodel(inputs)
                loss = criterion(predicts, targets)

                test_loss += loss.item() * targets.size(0)#累计总损失
                n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()#正确预测数量
                n_test += targets.size(0)#测试样总数

                ground_truth = targets
                predictions = torch.argmax(predicts, dim=1)
                TP += torch.logical_and(predictions.bool(), ground_truth.bool()).sum().item()
                FP += torch.logical_and(predictions.bool(), ~ground_truth.bool()).sum().item()
                FN += torch.logical_and(~predictions.bool(), ground_truth.bool()).sum().item()
                TN += torch.logical_and(~predictions.bool(), ~ground_truth.bool()).sum().item()
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        specificity = TN / (TN + FP)
        f1_score = 2 * precision * recall / (precision + recall)
        return  test_loss / n_test, n_correct / n_test,precision, recall, specificity, f1_score #返回平均损失和准确率
    
    def run(self):
        # Print the parameters of model
        # for name, layer in self.Mymodel.named_parameters(recurse=True):
        # print(name, layer.shape, sep=" ")
        #加载数据集和测试集的数据加载器
        train_dataloader,val_dataloader, test_dataloader = load_dataset(tokenizer=self.tokenizer,
                                                         train_batch_size=self.args.train_batch_size,
                                                         val_batch_size=self.args.val_batch_size,#新增验证集合
                                                         test_batch_size=self.args.test_batch_size,
                                                         model_name=self.args.model_name,
                                                         method_name=self.args.method_name,
                                                         workers=self.args.workers)
        _params = filter(lambda x: x.requires_grad, self.Mymodel.parameters())#过滤出需要计算梯度的参数
        criterion = nn.CrossEntropyLoss()#损失函数
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.weight_decay)#优化器

        l_test_acc, l_trloss,lvaloss,l_teloss, l_epo =[], [], [], [], []#这些列表用于记录每个epoch的训练和测试损失以及准确率。
        # Get the best_loss and the best_acc
        best_loss, best_acc = 0, 0 #最好的损失和精度
        for epoch in range(self.args.num_epoch):#训练epoll轮
            train_loss, train_acc = self._train(train_dataloader, criterion, optimizer)#训练
            val_loss, val_acc = self._val(val_dataloader, criterion)#验证集
            test_loss, test_acc, test_precision, test_recall, test_specificity, test_f1_score = self._test(test_dataloader, criterion)#测试集
            l_epo.append(epoch), l_test_acc.append(test_acc), l_trloss.append(train_loss),lvaloss.append(val_loss),l_teloss.append(test_loss)
            if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):#更新最佳测试准确率和最佳损失
                best_test_acc, best_test__loss = test_acc, test_loss
            #写入日志
            self.logger.info(
                '{}/{} - {:.2f}%'.format(epoch + 1, self.args.num_epoch, 100 * (epoch + 1) / self.args.num_epoch))
            self.logger.info('[train] loss: {:.4f}, acc: {:.2f}'.format(train_loss, train_acc * 100))
            self.logger.info('[val] loss: {:.4f}, acc: {:.2f}'.format(val_loss, val_acc * 100))
            self.logger.info('[test] loss: {:.4f}, acc: {:.2f}'.format(test_loss, test_acc * 100))
            #打印论文测试指标
            self.logger.info('[test] precision: {:.4f},recall: {:.4f},specificity: {:.4f},f1_score: {:.4f}'.format(test_precision, test_recall,test_specificity,test_f1_score))
        self.logger.info('best_test_loss: {:.4f}, best_test_acc: {:.2f}'.format(best_test_acc, best_test__loss * 100))
        self.logger.info('log saved: {}'.format(self.args.log_name))
        
        # 画图
        #画测试精度
        plt.plot(l_epo, l_test_acc)
        plt.ylabel('test-accuracy')
        plt.xlabel('epoch')
        plt.savefig('test_acc.png')
        #画测试损失
        plt.plot(l_epo, l_teloss)
        plt.ylabel('test-loss')
        plt.xlabel('epoch')
        plt.savefig('test_loss.png')
        #画训练损失
        plt.plot(l_epo, l_trloss)
        plt.ylabel('train-loss')
        plt.xlabel('epoch')
        plt.savefig('train_loss.png')
        #画验证损失
        plt.plot(l_epo, lvaloss)
        plt.ylabel('val-loss')
        plt.xlabel('epoch')
        plt.savefig('val_loss.png')



if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    nb = Niubility(args, logger)
    nb.run()

    #测量模型参数量
    # model = nb.Mymodel
    # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Number of trainable parameters: ", num_params)
    
