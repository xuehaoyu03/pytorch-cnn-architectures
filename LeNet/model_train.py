from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
from model import LeNet
import torch 
import torch.nn as nn
import copy
import time
import pandas as pd

# 加载数据集
def train_val_data_process():
    train_data = FashionMNIST(root='../data',
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=28),transforms.ToTensor()]),
                          download=True)

    train_data,val_data = Data.random_split(train_data,[round(0.8* len(train_data)),round(0.2* len(train_data))])

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=0)
    val_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=0)
    return train_dataloader,val_dataloader

train_dataloader,val_dataloader = train_val_data_process()

# 模型训练函数
def train_model_process(model,train_dataloader,val_dataloader,num_epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    #device = torch.device("cpu")
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高精确度
    best_acc = 0.0
    # 训练集损失函数列表
    train_loss_all = []
    # 验证集损失函数列表
    val_loss_all = []
    # 训练集精度列表
    train_acc_all = []
    # 验证集精度列表
    val_acc_all = []

    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch{}/{}".format(epoch,num_epochs-1))
        print("-"*10)

        # 初始化参数
        train_loss = 0.0
        train_corrects = 0.0

        val_loss = 0.0
        val_corrects = 0.0

        # 样本数量
        train_num = 0
        val_num = 0

        # 对每一个mini-batch训练和计算
        for step,(b_x,b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.train()

            output = model(b_x)
            # 查找概率最大的行标
            pre_lab = torch.argmax(output,dim=1)

            loss = criterion(output,b_y)

            # 将梯度置为0 
            optimizer.zero_grad()

            # 反向传播
            loss.backward()
            # 利用反向传播更新网络参数
            optimizer.step()

            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)

            # 如果预测正确，则准确度加一
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 累加样本
            train_num += b_x.size(0)
        
        # 对每一个mini-batch验证和计算
        for step,(b_x,b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 模型进行评估模式
            model.eval()

            output = model(b_x)

            # 查找概率最大的行标
            pre_lab = torch.argmax(output,dim=1)

            # 计算loss
            loss = criterion(output,b_y)

            # 对损失函数进行累加
            val_loss += loss.item() * b_x.size(0)

            # 如果预测正确，则准确度加一
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 累加样本
            val_num += b_x.size(0)

        # 计算每一个epoch的loss值和准确率
        train_loss_all.append(train_loss / train_num)
        

        train_acc_all.append((train_corrects.float() / train_num).cpu().item())
        val_acc_all.append((val_corrects.float() / val_num).cpu().item())



        val_loss_all.append(val_loss / val_num)
        

        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use // 60,time_use % 60))

    # 选择最优参数 保存模型
    torch.save(model.state_dict(best_model_wts),'LeNet/best_model.pth')


    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                        "train_loss_all":train_loss_all,
                                        "val_loss_all":val_loss_all,
                                        "train_acc_all":train_acc_all,
                                        "val_acc_all":val_acc_all,})

    return train_process
    
def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"],train_process.train_loss_all,'ro-',label="train_loss")
    plt.plot(train_process["epoch"],train_process.val_loss_all,'bs-',label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1,2,2)
    plt.plot(train_process["epoch"],train_process.train_acc_all,'ro-',label="train_acc")
    plt.plot(train_process["epoch"],train_process.val_acc_all,'bs-',label="val_acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.show()


if __name__=="__main__":
    # 模型实例化
    LeNet = LeNet()
    # 加载数据集
    train_dataloader,val_dataloader = train_val_data_process()
    # 训练模型
    train_process = train_model_process(LeNet,train_dataloader,val_dataloader,50)
    matplot_acc_loss(train_process)





