import argparse
import matplotlib.pyplot as plt
import os
from typing import Dict, List
import torch
from sklearn.metrics import accuracy_score
from glob import glob


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--devices', default="0,", type=str)
    parser.add_argument('--data_path', default="/media/data1/sangjunchung/dacon/object_classification")
    parser.add_argument('--model', default='resnet50')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--output_dir', default="/media/data1/sangjunchung/dacon/object_classification/checkpoint")
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--n_splits', default=5, type=int, help='kfold n_splits')
    parser.add_argument('--early_stopping', default=20, type=int)
    parser.add_argument('--kfold', action='store_true')
    return parser
