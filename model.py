import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from timm import create_model

import numpy as np


class Model(L.LightningModule):
    def __init__(self, model, num_classes, pretrained, output_dir):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(model, num_classes=num_classes, pretrained=pretrained)
        self.output_dir = output_dir
        self.epoch_train_loss = 0.0
        self.val_loss = []
        self.correct, self.total = 0, 0
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)
        self.log('train_loss', loss.item(), sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):        
        images, labels = batch
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)
        self.log("val_loss", loss.item(), sync_dist=True)
        self.val_loss.append(loss.item())
        
        predicted = torch.argmax(outputs, 1)
        self.correct += (predicted==labels).sum().item()
        self.total += len(labels)
        
        return loss
    
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)
        self.log('test_loss', loss.item(), on_epoch=True, sync_dist=True)
        return loss
    
    def on_validation_epoch_end(self):
        avg = np.mean(np.array(self.val_loss))
        self.log("acc", self.correct / self.total * 100, sync_dist=True)
        self.log("avg", avg, sync_dist=True)                
                        
        self.val_loss.clear()
        self.correct, self.total = 0, 0
        
        return avg
        
    # def on_test_epoch_end(self):
    #     self.log("test_loss", 12345)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters())
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100)
        
        return {"optimizer" : optimizer, "lr_scheduler" : scheduler}
    
    def configure_callbacks(self):
        tqdm_cb = TQDMProgressBar(refresh_rate=10)
        ckpt_cb = ModelCheckpoint(
            save_top_k = 1,
            monitor='val_loss',
            mode='min',
        	dirpath= self.output_dir,
            filename="{epoch:02d}_{acc:2f}"
        )
        early_cb = EarlyStopping(monitor='val_loss', mode='min', patience=10)
        return [tqdm_cb, ckpt_cb, early_cb]