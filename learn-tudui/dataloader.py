import torchvision
from sympy import false
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./dataset",download=True,train=false,transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data,
                         batch_size=4,
                         shuffle=True,
                         num_workers=0,
                         drop_last=False)

img, target = test_data[0]
print(img.shape) # torch.Size([3, 32, 32])
print(target)    # 3

writer = SummaryWriter("dataloader")
step = 0

for data in test_loader:
    imgs, targets = data
    # print(imgs.shape) # torch.Size([4, 3, 32, 32])
    # print(targets)    # tensor([6, 2, 6, 1])
    writer.add_images("test_data",imgs,step)
    step = step + 1

writer.close()