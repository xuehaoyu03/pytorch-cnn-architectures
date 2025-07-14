import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# 定义 Tudui 模型类
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

# 实例化模型
tudui = Tudui()
print(tudui)

# 创建输入张量
input = torch.ones((64, 3, 32, 32))
# 前向传播得到输出
output = tudui(input)
print(output)
print(output.shape)

# 使用 SummaryWriter 可视化计算图
writer = SummaryWriter("./logs_seq")
writer.add_graph(tudui, input)
writer.close()