import torch 
from torch import nn
import torch.backends
import torch.backends.mps
from torchinfo import summary

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.c1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2)
        self.sig = nn.Sigmoid()
        self.x2 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.c3 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2,stride=2)

        self.flatten = nn.Flatten()
        self.f5 = nn.Linear(5*5*16,120)
        self.f6 = nn.Linear(120,84)
        self.f7 = nn.Linear(84,10)

    def forward(self,x):
        # 卷积
        x = self.sig(self.c1(x))
        # 池化
        x = self.x2(x)
        # 卷积
        x = self.sig(self.c3(x))
        # 池化
        x = self.s4(x)
        # 全连接
        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x

if __name__ =="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    
    model = LeNet().to(device)
    print(summary(model, input_size=(1, 1, 28, 28), device=device))




