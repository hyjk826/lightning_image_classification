import cv2
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np


class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels = None, img_size = None , mode = None):    
        self.img_paths = img_paths
        self.labels = labels
        self.img_size = img_size
        self.mode = mode 
        
        if mode == 'train':
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Rotate(30, p = 1),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Normalize(),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(),
                ToTensorV2(),
            ])        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image = img)["image"]
        if self.mode == 'test':
            return img
        else:
            return img, self.labels[index]