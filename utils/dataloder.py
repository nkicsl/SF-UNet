import os
from PIL import Image, ImageFilter
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torchvision.transforms.functional as F
import torchvision.transforms as T

class SkinData(Dataset):
    def __init__(self, dataset_path, transforms=None, input_size = (224, 224)):
        super(SkinData, self).__init__()
        self.dataset_path = dataset_path
        self.transforms = transforms
        self.input_size = input_size

    def __getitem__(self, index):

        img_path = os.path.join(self.dataset_path, 'image')
        label_path = os.path.join(self.dataset_path, 'label')

        img_list = os.listdir(img_path)
        label_list = os.listdir(label_path)

        img_list.sort()
        label_list.sort()

        img = Image.open(os.path.join(img_path, img_list[index])).convert('RGB')
        label = Image.open(os.path.join(label_path, label_list[index])).convert('L')

        img = img.resize(size=self.input_size, resample=Image.BICUBIC)
        label = label.resize(size=self.input_size, resample=Image.NEAREST)

        if self.transforms is not None:
            img, label = self.transforms(img, label)

        return img, label

    def __len__(self):
        img_path = os.path.join(self.dataset_path, 'image')
        img_list = os.listdir(img_path)
        return len(img_list)