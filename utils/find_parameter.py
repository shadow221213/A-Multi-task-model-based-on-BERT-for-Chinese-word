import os

import torch
from tools import parse_args, set_seed
from transformers import AutoTokenizer

from MutiTask.train import mtl_train

if __name__ == '__main__':
    set_seed(221213)
    args = parse_args( )
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast=True)
    device = torch.device('cuda' if torch.cuda.is_available( ) else 'cpu')

    learning_rates = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6]
    weight_decay = [0.01, 0.05, 0.1]
    best_accuracy = 0
    best_lr = None
    best_wd = None
    out_dir = args.output_dir

    for lr in learning_rates:
        for wd in weight_decay:
            args.weight_decay = wd
            args.learning_rate = lr
            args.output_dir = os.path.join(out_dir, f'lr_{lr}', f'wd_{wd}')
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

            trainer = mtl_train(args, tokenizer, device)
            print(f"\n===== 测试学习率: {lr} =====")
            print(f"\n===== 测试weight_decay: {wd} =====")
            print(f"\n===== 输出路径：{args.output_dir} =====")
            trainer.train( )

            # 评估模型
            eval_metrics = trainer.evaluate( )
            current_acc = eval_metrics["eval_accuracy"]
            print(f"\n{eval_metrics}")

            if current_acc > best_accuracy:
                best_accuracy = current_acc
                best_lr = lr
                best_wd = wd

    print(f"最佳学习率: {best_lr}, 最佳weight_decay：{best_wd}，准确率: {best_accuracy}")
