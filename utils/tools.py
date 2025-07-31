import argparse
import os
import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True

def get_results(trainer):
    original_report_to = trainer.args.report_to
    trainer.args.report_to = []
    eval_results = trainer.evaluate( )
    trainer.args.report_to = original_report_to

    return eval_results

# 设置命令行参数解析器
def parse_args( ):
    parser = argparse.ArgumentParser(description="Multi-task Learning for NLP")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for AdamW optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay for AdamW optimizer')
    parser.add_argument('--pretrained_model', type=str, default='bert-base-chinese', help='Pretrained BERT model')
    parser.add_argument('--train_data', type=str, default='./data/train.csv', help='Path to training data')
    parser.add_argument('--test_data', type=str, default='./data/test.csv', help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save model checkpoints')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--unfreeze_epoch', type=int, default=3, help='Epoch to unfreeze cls layers')
    parser.add_argument('--grad_accumulation_steps', type=int, default=4, help='Gradient Accumulation steps for training')
    parser.add_argument('--freeze_cls', action='store_true', help='Freeze classification layers initially')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--train', action='store_true', help='Enable train model')
    parser.add_argument('--evaluate', action='store_true', help='Enable evaluate model')
    parser.add_argument('--mtl', action='store_true', help='Enable MTL model')
    parser.add_argument('--seg', action='store_true', help='Enable Segmentation model')
    parser.add_argument('--cls', action='store_true', help='Enable Classification model')
    parser.add_argument('--ner', action='store_true', help='Enable NER model')

    return parser.parse_args( )

def parse_args_aug( ):
    parser = argparse.ArgumentParser(description="preprocess data")
    parser.add_argument('--chunk_size', type=int, default=128, help='Batch size for preprocessing')
    parser.add_argument('--input_csv', type=str, default='./data/train.csv', help='Path to input data')
    parser.add_argument('--output_csv', type=str, default='./data/train_augment1.csv', help='Path to output data')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save model checkpoints')
    parser.add_argument('--temp_dir', type=str, default='./data/temp', help='Directory to save temp data')
    parser.add_argument('--pretrained_model', type=str, default='bert-base-chinese', help='Pretrained BERT model')
    parser.add_argument('--delete', action='store_true', help='Enable delete temp files')

    return parser.parse_args( )