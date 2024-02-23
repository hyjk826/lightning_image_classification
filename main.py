from args_parser import get_args_parser
from model import Model
from datamodule import DataModule

import lightning as L
from lightning.pytorch.loggers import CSVLogger

import torch
import os
import utils
import numpy as np
from custom_dataset import Custom_dataset

from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold

def main(args):
    
    torch.set_float32_matmul_precision('medium')
    
    logger = CSVLogger("logs", name = f"{args.model}")
    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.devices,
        max_epochs=args.epochs,
        logger=logger,
        log_every_n_steps = 30,
    )
    
    if args.kfold:
        
        label_encoder = utils.label_encoder(args.data_path + '/train')        
        img_paths, train_labels = utils.load_img_paths(args.data_path + '/train')
    
        img_paths = np.array(img_paths)
        train_labels = np.array([label_encoder[i] for i in train_labels])
        
        cv = StratifiedKFold(n_splits=args.n_splits, random_state=args.seed, shuffle=True)
    
        for fold, (train_idx, val_idx) in enumerate(cv.split(img_paths, train_labels)):
            X_train_path, X_val_path = img_paths[train_idx], img_paths[val_idx]
            y_train, y_val = train_labels[train_idx], train_labels[val_idx]
            train_dataset = Custom_dataset(img_paths=X_train_path, labels=y_train, img_size=224, mode='train')
            val_dataset = Custom_dataset(img_paths=X_val_path, labels=y_val, img_size=224, mode='val')

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)            
            
            os.makedirs(f"{args.output_dir}/fold{fold}", exist_ok=True)
            model = Model(model=args.model, num_classes=args.num_classes, pretrained=args.pretrained, output_dir = f"{args.output_dir}/fold{fold}", early_stopping=args.early_stopping)                    
            
            logger = CSVLogger("logs", name = f"{args.model}")
            trainer = L.Trainer(
                accelerator="gpu",
                devices=args.devices,
                max_epochs=args.epochs,
                logger=logger,
                log_every_n_steps = 30,
            )       
            trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)        
            
            return     
    
    
    
    os.makedirs(args.output_dir, exist_ok=True)    
    model = Model(model=args.model, num_classes=args.num_classes, pretrained=args.pretrained, output_dir = args.output_dir, early_stopping=args.early_stopping)                    
    datamodule = DataModule(data_path = args.data_path, batch_size=args.batch_size)    
    
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)