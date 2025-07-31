import gc

import torch
from transformers import AutoTokenizer

from MutiTask.train import mtl_train
from SingleTask.train import stl_train
from utils.tools import get_results, parse_args, set_seed

if __name__ == "__main__":
    set_seed(221213)
    args = parse_args( )
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast=True)
    device = torch.device('cuda' if torch.cuda.is_available( ) else 'cpu')

    if args.mtl:
        mtl_trainer = mtl_train(args, tokenizer, device, is_final_eval=False)
        if args.train:
            print("开始训练mtl模型")
            mtl_trainer.train( )

        if 'mtl_trainer' in locals( ):
            del mtl_trainer
        gc.collect( )
        if torch.cuda.is_available( ):
            torch.cuda.empty_cache( )

    if args.seg:
        seg_trainer = stl_train("seg", args, tokenizer, device)
        if args.train:
            print("开始训练seg模型")
            seg_trainer.train( )

        if 'seg_trainer' in locals( ):
            del seg_trainer
        gc.collect( )
        if torch.cuda.is_available( ):
            torch.cuda.empty_cache( )

    if args.cls:
        cls_trainer = stl_train("cls", args, tokenizer, device)
        if args.train:
            print("开始训练cls模型")
            cls_trainer.train( )

        if 'cls_trainer' in locals( ):
            del cls_trainer
        gc.collect( )
        if torch.cuda.is_available( ):
            torch.cuda.empty_cache( )

    if args.ner:
        ner_trainer = stl_train("ner", args, tokenizer, device)
        if args.train:
            print("开始训练ner模型")
            ner_trainer.train( )

        if 'ner_trainer' in locals( ):
            del ner_trainer
        gc.collect( )
        if torch.cuda.is_available( ):
            torch.cuda.empty_cache( )

    if args.evaluate:
        args.freeze = False
        if args.mtl:
            mtl_trainer = mtl_train(args, tokenizer, device, is_final_eval=True)
            eval_mtl = get_results(mtl_trainer)
            print(f'多任务模型测试结果：{eval_mtl}')
        if args.seg:
            seg_trainer = stl_train("seg", args, tokenizer, device)
            eval_seg = get_results(seg_trainer)
            print(f'seg模型测试结果：{eval_seg}')
        if args.cls:
            cls_trainer = stl_train("cls", args, tokenizer, device)
            eval_cls = get_results(cls_trainer)
            print(f'cls模型测试结果：{eval_cls}')
        if args.ner:
            ner_trainer = stl_train("ner", args, tokenizer, device)
            eval_ner = get_results(ner_trainer)
            print(f'ner模型测试结果：{eval_ner}')

            print(f'cls模型测试结果：{eval_cls}')
        if args.ner:
            ner_trainer = stl_train("ner", args, tokenizer, device)
            eval_ner = get_results(ner_trainer)
            print(f'ner模型测试结果：{eval_ner}')
