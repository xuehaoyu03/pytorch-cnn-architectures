import torch
from torch import nn
from torchsummary import summary


class Inception(nn.Module):
    def __init__(self,in_channels,c1,c2,c3,c4):
        super(Inception,self).__init__()
        self.ReLU = nn.ReLU()

        # 路线1: 1X1卷积 c1:卷积核的数量
        self.p1_1 = nn.Conv2d(in_channels=in_channels,kernel_size=1,out_channels=c1)

        # 路线2: 1X1卷积,3X3卷积 c2：两个卷积操作的卷积核的数量
        self.p2_1 = nn.Conv2d(in_channels=in_channels, kernel_size=1, out_channels=c2[0])
        self.p2_2 = nn.Conv2d(in_channels=c2[0], kernel_size=3, out_channels=c2[1],padding=1)

        # 路线3: 1X1卷积,5X5卷积 c3：两个卷积操作的卷积核的数量
        self.p3_1 = nn.Conv2d(in_channels=in_channels, kernel_size=1, out_channels=c3[0])
        self.p3_2 = nn.Conv2d(in_channels=c3[0], kernel_size=5, out_channels=c3[1], padding=2)

        # 路线4: 3X3最大池化,1X1卷积 c4：两个卷积操作的卷积核的数量
        self.p4_1 = nn.MaxPool2d(kernel_size=3,padding=1,stride=1)
        self.p4_2 = nn.Conv2d(in_channels=in_channels, kernel_size=1, out_channels=c4)

    def forward(self,x):
        p1 = self.ReLU(self.p1_1(x))
        p2 = self.ReLU(self.p2_2(self.ReLU(self.p2_1(x))))
        p3 = self.ReLU(self.p3_2(self.ReLU(self.p3_1(x))))
        p4 = self.ReLU(self.p4_2(self.p4_1(x)))

        #print(p1.shape,p2.shape,p3.shape,p4.shape)

        # dim=1是把[N(批量大小),C(输入通道数),H(高度),W(宽度)]中的第二个C里的所有通道数进行拼接
        return torch.cat((p1,p2,p3,p4),dim=1)


# 辅助分类器
class AuxClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxClassifier, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GoogLeNet(nn.Module):
    def __init__(self,Inception,num_classes=10, aux_logits=True):
        super(GoogLeNet,self).__init__()
        self.aux_logits = aux_logits

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b3 = nn.Sequential(
            Inception(192,64,(96,128),(16,32),32),
            Inception(256,128,(128,192),(32,96),64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (128, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b5 = nn.Sequential(
            Inception(832,256,(160,320),(32,128),128),
            Inception(832,384,(128,384),(48,128),128),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024,10)
        )

        if self.aux_logits:
            self.aux1 = AuxClassifier(512, num_classes)  # 对应4a模块输出
            self.aux2 = AuxClassifier(528, num_classes)  # 对应4d模块输出

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)


    def forward(self,x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)

        # 经过4a模块后
        x = self.b4[0](x)
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        else:
            aux1 = None

        # 经过4d模块后
        x = self.b4[1](x)
        x = self.b4[2](x)
        x = self.b4[3](x)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
        else:
            aux2 = None

        # 剩余层
        x = self.b4[4:](x)
        x = self.b5(x)
        return x

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GoogLeNet(Inception).to(device)
    print(summary(model, input_size=(1,224,224)))



































