from args_parser import get_args_parser
from model import Model
from datamodule import DataModule

import lightning as L
from lightning.pytorch.loggers import CSVLogger

import torch
import os

def main(args):
    torch.set_float32_matmul_precision('medium')
    os.makedirs(args.output_dir, exist_ok=True)    
    model = Model(model=args.model, num_classes=args.num_classes, pretrained=args.pretrained, output_dir = args.output_dir)
    datamodule = DataModule(data_path = args.data_path, batch_size=args.batch_size)
    
    logger = CSVLogger("logs", name = f"{args.model}")
    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.devices,
        max_epochs=args.epochs,
        logger=logger,
        log_every_n_steps = 1,
        enable_checkpointing=False
    )
    
    trainer.fit(model=model, datamodule=datamodule, ckpt_path='/media/data1/sangjunchung/dacon/object_classification/checkpoint/epoch=01_val_loss=0.429568.ckpt')


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)