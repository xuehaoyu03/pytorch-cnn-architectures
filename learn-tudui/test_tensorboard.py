import numpy as np
from PIL import Image

from torch.utils.tensorboard import SummaryWriter


Writer = SummaryWriter("logs")
image_path = "data/train/apple/0000.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

# 图像类
Writer.add_image("test",img_array,1,dataformats='HWC')

# y = 2x
for i in  range(100):
    # 数字类
    Writer.add_scalar("y=2x",2 * i,i)

Writer.close()