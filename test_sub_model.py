'''
Descripttion: 
version: 
Date: 2021-12-29 21:03:16
LastEditTime: 2021-12-30 00:12:59
'''
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import math
import numpy as np


class Layer1(nn.Module):
    def __init__(self, n_input, n_output): # 控制输入输出的维度
        super(Layer1, self).__init__() # 需要和SimpleModel一致不可以为nn.Module 或者Module
        self.n_input = n_input
        self.n_output = n_output
        self.W2 = Layer2(1, 1)
        self.W = nn.Linear(n_input, n_output, bias=False)

        self.init()
    def init(self): # 手动初始化w1 w2 w3 全为1
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(1., 1.)
    def forward(self, x):
        y  = self.W(x)
        y2 = self.W2(y)
        return y2

class Layer2(nn.Module):
    def __init__(self, n_input, n_output): # 控制输入输出的维度
        super(Layer2, self).__init__() # 需要和SimpleModel一致不可以为nn.Module 或者Module
        self.n_input = n_input
        self.n_output = n_output
        self.W = nn.Embedding(n_input, n_output)
        self.init()
    def init(self): # 手动初始化w1 w2 w3 全为1
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(1., 1.)
    def forward(self, x):
        y=  torch.matmul(self.W(torch.tensor(0)),x) # 取出Embeding的对应数据
        return y

model = Layer1(3,1)
lr=0.05
epoch=20
target = torch.tensor(5.)

# L2惩罚项，注意Adam并不是严格按照梯度下降更新的，如果严格按照梯度更新可以换成SGD优化器
optimizer = optim.SGD(model.parameters() , lr=lr)
for name, param in model.named_parameters(): #查看可优化的参数有哪些
    if param.requires_grad:
        print("name",name)
print(model)
input_x = torch.rand(1,3)

# 查看第一层模型的参数
# for i in range(epoch):
#     optimizer.zero_grad() # 梯度清零
#     result = model(input_x)
#     loss_train = (5.0-result)**2
#     print("loss_train:",loss_train) # tensor([1.2887], grad_fn=<PowBackward0>)
#     print("before backward weight.data:",model.W.weight.data)  #  可以查看线性转换矩阵的权重
#     loss_train.backward() # 反向传播计算得到每个参数的梯度值
#     print("after backward weight.data:",model.W.weight.data)  #  可以查看线性转换矩阵的权重
#     print("weight.grad:", model.W.weight.grad)
#     print("weight - weight.grad*lr:", model.W.weight - model.W.weight.grad*lr)
#     optimizer.step() # 梯度下降执行一步参数更新
#     print("affter step weight.data:", model.W.weight.data)

# 查看第二层模型的参数
for i in range(epoch):
    optimizer.zero_grad() # 梯度清零
    result = model(input_x)
    loss_train = (5.0-result)**2
    print("loss_train:",loss_train) # tensor([1.2887], grad_fn=<PowBackward0>)
    print("before backward weight.data:",model.W2.W.weight.data)  #  可以查看线性转换矩阵的权重
    loss_train.backward() # 反向传播计算得到每个参数的梯度值
    print("after backward weight.data:",model.W2.W.weight.data)  #  可以查看线性转换矩阵的权重
    print("weight.grad:", model.W2.W.weight.grad)
    print("weight - weight.grad*lr:", model.W2.W.weight.data - model.W2.W.weight.grad*lr)
    optimizer.step() # 梯度下降执行一步参数更新
    print("affter step weight.data:", model.W2.W.weight.data)
