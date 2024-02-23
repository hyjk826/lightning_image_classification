import torch
import timm
from seed import init_seed
from custom_dataset import Custom_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import pandas as pd
from glob import glob
import numpy as np
import utils
from glob import glob
from model import Model
from lightning.pytorch.loggers import CSVLogger
import lightning as L
from args_parser import get_args_parser


def main(args):
    init_seed(0)
    
    torch.set_float32_matmul_precision('medium')
    
    test_img_paths = sorted(glob(args.data_path + '/test/*'))
        
    test_dataset = Custom_dataset(test_img_paths, labels=None, img_size=224, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    folder = args.data_path + '/checkpoint'
    
    pred_ensemble = []
    
    with torch.no_grad():
        for fold in np.arange(args.n_splits):
            model_path = glob(f"{folder}/fold{fold}/*")
            logger = CSVLogger("logs", name = f"{args.model}")
            trainer = L.Trainer(
                accelerator="gpu",
                devices=args.devices,
                max_epochs=args.epochs,
                logger=logger,
                log_every_n_steps = 30,
            )       
            model = Model(model=args.model, num_classes=args.num_classes, pretrained=args.pretrained, output_dir = args.output_dir, early_stopping=args.early_stopping)                    
            pred = trainer.predict(model=model, dataloaders=test_loader, ckpt_path=model_path[0])
            
            pred_prob = []        
            
            for batch in pred:
                
                pred_prob.extend(batch.detach().cpu().numpy())
                
            pred_ensemble.append(pred_prob)
    
    
    
    pred = np.sum(pred_ensemble, axis=0)
    
    
    f_pred = np.argmax(pred, axis=-1).tolist()
    
    test_df = pd.read_csv(args.data_path + '/sample_submission.csv')
    
    label_encoder = utils.label_encoder(args.data_path + '/train')
    label_decoder = utils.label_decoder(args.data_path + '/train')
    
    pred = [label_decoder[i] for i in f_pred]
    
    test_df['target'] = pred
    test_df.to_csv(f'submission_{args.model}_kfold{args.n_plits}.csv', index=False)
    
    print(test_df)
    
    
    

if __name__ == "__main__":    
    args = get_args_parser().parse_args()
    main(args)
    