from args_parser import get_args_parser
from model import Model
from datamodule import DataModule, KFoldDataModule

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
            
    if args.kfold:        

        for fold in range(args.n_splits):
            logger = CSVLogger("logs", name = f"{args.model}")
            trainer = L.Trainer(
                accelerator="gpu",
                devices=args.devices,
                max_epochs=args.epochs,
                logger=logger,
                log_every_n_steps = 30,
            )
            os.makedirs(f"{args.output_dir}/{args.model}/fold{fold}", exist_ok=True)
            model = Model(model=args.model, num_classes=args.num_classes, pretrained=args.pretrained, output_dir = f"{args.output_dir}/{args.model}/fold{fold}", early_stopping=args.early_stopping)                 
            datamodule = KFoldDataModule(data_path=args.data_path, batch_size=args.batch_size, n_splits=args.n_splits, k = fold, seed=args.seed)
            trainer.fit(model=model, datamodule=datamodule)          
        
        return
    
    
    logger = CSVLogger("logs", name = f"{args.model}")
    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.devices,
        max_epochs=args.epochs,
        logger=logger,
        log_every_n_steps = 30,
    )
    
    os.makedirs(args.output_dir, exist_ok=True)    
    model = Model(model=args.model, num_classes=args.num_classes, pretrained=args.pretrained, output_dir = args.output_dir, early_stopping=args.early_stopping)                    
    datamodule = DataModule(data_path = args.data_path, batch_size=args.batch_size)    
    
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)