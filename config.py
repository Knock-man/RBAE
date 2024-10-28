#配置文件
import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime

import torch

#获取配置
def get_config():
    parser = argparse.ArgumentParser() #创建命令行参数解析器
    '''Base'''

    parser.add_argument('--num_classes', type=int, default=2)#类别数量
    parser.add_argument('--model_name', type=str, default='bert',#预处理模型
                        choices=['bert', 'roberta'])
    parser.add_argument('--method_name', type=str, default='lstm_textcnn_attention',#下游模型
                        choices=['gru', 'rnn', 'bilstm', 'lstm', 'fnn', 'textcnn', 'attention', 'lstm+textcnn',
                                 'lstm_textcnn_attention'])

    '''Optimization'''
    parser.add_argument('--train_batch_size', type=int, default=16)#训练每个批次样本数量
    parser.add_argument('--val_batch_size', type=int, default=8)#验证每个批次样本数量
    parser.add_argument('--test_batch_size', type=int, default=8)#测试每个批次样本数量
    parser.add_argument('--num_epoch', type=int, default=1)#训练的周期数de
    parser.add_argument('--lr', type=float, default=1e-5)#训练模型的学习率
    parser.add_argument('--weight_decay', type=float, default=0.01)#正则项的衰退系数(λ的值)

    '''Environment'''
    parser.add_argument('--device', type=str, default='cpu')#运行设备
    parser.add_argument('--backend', default=False, action='store_true')
    parser.add_argument('--workers', type=int, default=0)#进程并行的工作量
    parser.add_argument('--timestamp', type=int, default='{:.0f}{:03}'.format(time.time(), random.randint(0, 999)))#生成时间戳

    args = parser.parse_args()#解析命令行输入的参数
    args.device = torch.device(args.device)#将字符串形式的设备参数转换为 PyTorch 设备对象

    '''logger'''#日志 记录每次运行的配置和结果
    args.log_name = '{}_{}_{}.log'.format(args.model_name, args.method_name,#日志名称
                                          datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
    if not os.path.exists('logs'):#保证日志目录存在
        os.mkdir('logs')
    logger = logging.getLogger()#获得日志对象
    logger.setLevel(logging.INFO)#设置日志级别
    logger.addHandler(logging.StreamHandler(sys.stdout))#将日志消息输出到标准输出流
    logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))#将日志消息写入到文件
    return args, logger
