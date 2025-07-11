from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms

image_path = "data/train/apple/0000.jpg"
img = Image.open(image_path)

Writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)

Writer.add_image("trans_img",tensor_img)
Writer.close()

