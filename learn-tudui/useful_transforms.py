from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

Writer = SummaryWriter("logs")
image_path = "data/train/apple/0000.jpg"
img = Image.open(image_path)


# ToTensor
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)


Writer.add_image("to_tensor",img_tensor)


# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
Writer.add_image("Normalize",img_norm)

# resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = trans_tensor(img_resize)
print(img_resize)
Writer.add_image("resize",img_resize,0)

# compose
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2,trans_tensor])
ims_resize_2 = trans_compose(img)
Writer.add_image("resize",ims_resize_2,1)

Writer.close()

