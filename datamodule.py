import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from timm import create_model
        
import albumentations as A        
import utils

from sklearn.model_selection import train_test_split
from custom_dataset import Custom_dataset

class DataModule(L.LightningDataModule):
    def __init__(self, data_path, batch_size, seed = 0):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.seed = seed
    
    def setup(self, stage=None):
        label_encoder = utils.label_encoder(self.data_path + '/train')
        img_paths, train_labels = utils.load_img_paths(self.data_path + '/train')
    
        img_paths = np.array(img_paths)
        train_labels = np.array([label_encoder[i] for i in train_labels])
        
        X_train, X_val, y_train, y_val = train_test_split(img_paths, train_labels, random_state=self.seed, test_size=0.2, stratify=train_labels)

        self.train_dataset = Custom_dataset(img_paths=X_train, labels=y_train, img_size=224, mode='train')
        self.val_dataset = Custom_dataset(img_paths=X_val, labels=y_val, img_size=224, mode='val')                        
                        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
     
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)