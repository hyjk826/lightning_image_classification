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

from sklearn.model_selection import train_test_split, StratifiedKFold
from custom_dataset import Custom_dataset

class DataModule(L.LightningDataModule):
    def __init__(self, data_path, batch_size, seed = 0):
        super().__init__()
        self.save_hyperparameters()
    
    def setup(self, stage=None):
        label_encoder = utils.label_encoder(self.hparams.data_path + '/train')
        img_paths, train_labels = utils.load_img_paths(self.hparams.data_path + '/train')
    
        img_paths = np.array(img_paths)
        train_labels = np.array([label_encoder[i] for i in train_labels])
        
        X_train, X_val, y_train, y_val = train_test_split(img_paths, train_labels, random_state=self.hparams.seed, test_size=0.2, stratify=train_labels)

        self.train_dataset = Custom_dataset(img_paths=X_train, labels=y_train, img_size=224, mode='train')
        self.val_dataset = Custom_dataset(img_paths=X_val, labels=y_val, img_size=224, mode='val')                        
                        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, pin_memory=True)
     
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True)
    

class KFoldDataModule(L.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int, n_splits: int=5, k: int=0, seed: int=0):
        super().__init__()
        self.save_hyperparameters()
    
    def setup(self, stage=None):
        label_encoder = utils.label_encoder(self.hparams.data_path + '/train')
        img_paths, train_labels = utils.load_img_paths(self.hparams.data_path + '/train')
    
        img_paths = np.array(img_paths)
        train_labels = np.array([label_encoder[i] for i in train_labels])
        
        skf = StratifiedKFold(n_splits=self.hparams.n_splits, shuffle=True, random_state=self.hparams.seed)
        
        all_splits = [i for i in skf.split(img_paths, train_labels)]
        
        X_train, X_val = img_paths[all_splits[self.hparams.k][0]], img_paths[all_splits[self.hparams.k][1]]
        y_train, y_val = train_labels[all_splits[self.hparams.k][0]], train_labels[all_splits[self.hparams.k][1]]        

        self.train_dataset = Custom_dataset(img_paths=X_train, labels=y_train, img_size=224, mode='train')
        self.val_dataset = Custom_dataset(img_paths=X_val, labels=y_val, img_size=224, mode='val')                        
                        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, pin_memory=True)
     
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True)