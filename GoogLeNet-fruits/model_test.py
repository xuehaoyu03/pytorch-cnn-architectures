import torch
import torch.utils.data as Data
from torchvision.datasets import ImageFolder
from model import GoogLeNet,Inception
from torchvision import transforms
from PIL import Image


# 加载数据集
def test_data_process():
    ROOT_TRAIN = r'data/test'

    # 进行正态分布归一化：转换成标准正态分布，使得数据处于激活函数梯度最大的区间
    normalize = transforms.Normalize([0.22890999,0.1963964 ,0.14335695], [0.09950233,0.07996743,0.06593084])

    # 定义数据集处理方法变量
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    # 加载数据集
    test_data = ImageFolder(ROOT_TRAIN, transform=test_transform)


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
    model = GoogLeNet(Inception)
    model.load_state_dict(torch.load('/Users/xuehaoyu/Desktop/learnAI/pytorch_pao/code/GoogLeNet-fruits/best_model.pth'))

    # 加载数据集
    test_dataloader = test_data_process()

    test_model_process(model,test_dataloader)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = model.to(device)

    classes = ['apple', 'banana','grape','orange','pear']
    # with torch.no_grad():
    #     for b_x,b_y in test_dataloader:
    #         b_x = b_x.to(device)
    #         b_y = b_y.to(device)
    #
    #         #设置模型为验证模式
    #         model.eval()
    #         output = model(b_x)
    #
    #         pre_lab = torch.argmax(output,dim=1)
    #         result = pre_lab.item()
    #         label = b_y.item()
    #
    #         print("预测值：",classes[result],"-----","真实值：",classes[label])

    image = Image.open('113.png').convert('RGB')
    normalize = transforms.Normalize([0.162, 0.151, 0.138], [0.058, 0.052, 0.048])

    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    image = test_transform(image)

    # 增加批次维度
    image = image.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        image = image.to(device)
        output = model(image)
        pre_lab = torch.argmax(output, dim=1)
        result = pre_lab.item()

    print(classes[result])







