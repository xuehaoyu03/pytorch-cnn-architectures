from torch.utils.data import Dataset
import cv2
from PIL  import Image
import os

class MyData(Dataset):

    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, item):
        img_name = self.img_path[item]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)

root_dir = "data/train"
apple_dir = "apple"
fruit_dataset = MyData(root_dir,apple_dir)
img,label = fruit_dataset[0]
# img.show()
print(len(fruit_dataset))
