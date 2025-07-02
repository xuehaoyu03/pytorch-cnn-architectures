import torch
import torch.utils.data as Data
from torchvision.datasets import FashionMNIST
from model import VGG16
from torchvision import transforms


# 加载数据集
def test_data_process():
    test_data = FashionMNIST(root='../data',
                          train=False,
                          transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),
                          download=True)


    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)
    
    return test_dataloader

def test_model_process(model,test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = model.to(device)

    test_corrects = 0.0
    test_num = 0

    # 推理过程中只有前向传播，不计算梯度
    with torch.no_grad():
        for test_data_x,test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            # 进行推理
            model.eval()
            output = model(test_data_x)

            pre_lab = torch.argmax(output,dim=1)

            test_corrects += torch.sum(pre_lab == test_data_y.data)
            test_num += test_data_x.size(0)

    test_acc = (test_corrects.float() / test_num).cpu().item()
    print("test acc =",test_acc)
    print("----------------------------")


if __name__=="__main__":
    model = VGG16()
    model.load_state_dict(torch.load('/Users/xuehaoyu/Desktop/learn/pytorch_pao/code/VGG-16/best_model.pth'))

    # 加载数据集
    test_dataloader = test_data_process()

    test_model_process(model,test_dataloader)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = model.to(device)

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    with torch.no_grad():
        for b_x,b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            #设置模型为验证模式
            model.eval()
            output = model(b_x)

            pre_lab = torch.argmax(output,dim=1)
            result = pre_lab.item()
            label = b_y.item()

            print("预测值：",classes[result],"-----","真实值：",classes[label])
            








