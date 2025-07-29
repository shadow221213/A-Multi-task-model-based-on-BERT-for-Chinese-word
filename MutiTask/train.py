import os

import torch
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from transformers import (AutoTokenizer, EarlyStoppingCallback, EvalPrediction,
                          Trainer, TrainerCallback, TrainingArguments,
                          get_linear_schedule_with_warmup)

from data.preprocess import TextDataset
from MutiTask.loss import MultiTaskLoss
from MutiTask.model import MultiTaskModel, MultiTaskModelConfig
from utils.tools import parse_args
import numpy as np
from termcolor import colored
from functools import partial
from sklearn.metrics import classification_report, f1_score, recall_score

def collate_fn( batch ):
    return {
        'input_ids':      torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'seg_labels':     torch.stack([x['seg_labels'] for x in batch]),
        'cls_labels':     torch.stack([x['cls_labels'] for x in batch]),
        'ner_labels':     torch.stack([x['ner_labels'] for x in batch]),
        # 'raw_text':       [x['raw_text'] for x in batch]
        }

class MultiTaskTrainer(Trainer):
    def __init__( self, *args, **kwargs ):
        super( ).__init__(*args, **kwargs)
        self.criterion = MultiTaskLoss( )
        self.best_metric = -1

    def compute_loss( self, model, inputs, return_outputs=False ):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        seg_labels = inputs['seg_labels']
        cls_labels = inputs['cls_labels']
        ner_labels = inputs['ner_labels']

        outputs = model(input_ids, attention_mask)
        seg_logits = outputs["seg_logits"]
        cls_logits = outputs["cls_logits"]
        ner_logits = outputs["ner_logits"]

        loss = self.criterion(
            seg_logits, seg_labels.long( ),
            cls_logits, cls_labels.long( ),
            ner_logits, ner_labels.long( )
            )

        return (loss, outputs) if return_outputs else loss

    def evaluate( self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval" ):
        eval_metrics = super( ).evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        current_accuracy = eval_metrics[f"{metric_key_prefix}_accuracy"]
        if current_accuracy > self.best_metric and self.args.report_to != []:
            self.best_metric = current_accuracy
            output_dir = f"{self.args.output_dir}/checkpoint-best"
            self.save_model(output_dir)
            print(f"\n===== 保存最佳模型到 {output_dir}，准确率: {self.best_metric} =====\n")

        return eval_metrics

class UnfreezeCallback(TrainerCallback):
    def __init__( self, freeze, unfreeze_epoch=5 ):
        self.unfreeze_epoch = unfreeze_epoch
        self.flag = True if freeze else False

    def on_epoch_begin( self, args, state, control, **kwargs ):
        if self.flag and state.epoch >= self.unfreeze_epoch:
            self.flag = False
            print("已解冻分类层参数")

            model = kwargs['model']
            # 解冻分类层和共享编码层
            for param in model.cls_fc.parameters( ):
                param.requires_grad = True
            for layer in model.encoder.bert.encoder.layer[:6]:
                for param in layer.parameters( ):
                    param.requires_grad = True

def mtl_compute_metrics(eval_pred: EvalPrediction, visualize: bool = False):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    seg_pred, cls_pred, ner_pred = predictions
    seg_labels, cls_labels, ner_labels = labels

    # 过滤掉填充标签-100
    def get_active_flat_labels_and_preds(logits, labels):
        preds_flat = logits.argmax(-1).flatten()
        labels_flat = labels.flatten()
        mask = labels_flat != -100
        return preds_flat[mask], labels_flat[mask]

    # 计算 seg 任务指标
    seg_preds_active, seg_labels_active = get_active_flat_labels_and_preds(seg_pred, seg_labels)
    seg_acc = accuracy_score(seg_labels_active, seg_preds_active)
    seg_f1 = f1_score(seg_labels_active, seg_preds_active, average='macro', zero_division=0)
    seg_recall = recall_score(seg_labels_active, seg_preds_active, average='macro', zero_division=0)

    # 计算 cls 任务指标
    cls_preds_active = cls_pred.argmax(-1)
    cls_acc = accuracy_score(cls_labels, cls_preds_active)
    cls_f1 = f1_score(cls_labels, cls_preds_active, average='macro', zero_division=0)
    cls_recall = recall_score(cls_labels, cls_preds_active, average='macro', zero_division=0)

    # 计算 ner 任务指标
    ner_preds_active, ner_labels_active = get_active_flat_labels_and_preds(ner_pred, ner_labels)
    ner_acc = accuracy_score(ner_labels_active, ner_preds_active)
    ner_f1 = f1_score(ner_labels_active, ner_preds_active, average='macro', zero_division=0)
    ner_recall = recall_score(ner_labels_active, ner_preds_active, average='macro', zero_division=0)

    avg_acc = (seg_acc + cls_acc + ner_acc) / 3

    # 计算损失 (这部分仍然需要在Tensor上进行)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    seg_loss = loss_fn(torch.from_numpy(seg_pred).view(-1, seg_pred.shape[-1]), torch.from_numpy(seg_labels).view(-1)).item()
    cls_loss = loss_fn(torch.from_numpy(cls_pred).view(-1, cls_pred.shape[-1]), torch.from_numpy(cls_labels).view(-1)).item()
    ner_loss = loss_fn(torch.from_numpy(ner_pred).view(-1, ner_pred.shape[-1]), torch.from_numpy(ner_labels).view(-1)).item()
    total_loss = seg_loss + cls_loss + ner_loss

    if visualize:
        seg_label_map = {'B': 0, 'I': 1, 'O': 2}
        cls_label_map = {'体育': 0, '娱乐': 1, '彩票': 2, '房产': 3, '教育': 4, '时政': 5, '游戏': 6, '社会': 7, '财经': 8}
        ner_label_map = {'O': 0, 'B-name': 1, 'I-name': 2, 'B-organization': 3, 'I-organization': 4, 'B-address': 5, 'I-address': 6, 'B-government': 7, 'I-government': 8, 'B-scene': 9, 'I-scene': 10, 'B-game': 11, 'I-game': 12, 'B-position': 13, 'I-position': 14, 'B-book': 15, 'I-book': 16, 'B-company': 17, 'I-company': 18, 'B-movie': 19, 'I-movie': 20}

        def visualize_sequence(seg_true, seg_pred, ner_true, ner_pred, seg_label_map, ner_label_map):
            text = "这也是张路老师在新浪的足彩推荐中第四次命中头奖！"
            seg_inv_map = {v: k for k, v in seg_label_map.items()}
            ner_inv_map = {v: k for k, v in ner_label_map.items()}

            print("\n" + "="*20 + " 预测结果可视化 " + "="*20)
            print("=== 原始文本 ===")
            print(text)
            print("\n=== 分词标签对比 ===")
            seg_tags = []
            for c, t, p in zip(text, seg_true, seg_pred):
                seg_t = seg_inv_map.get(t.item(), '?') if isinstance(t, torch.Tensor) else seg_inv_map.get(t, '?')
                seg_p = seg_inv_map.get(p.item(), '?') if isinstance(p, torch.Tensor) else seg_inv_map.get(p, '?')
                color = 'green' if t == p else 'red'
                seg_tags.append(colored(f"{c}({seg_t}->{seg_p})", color))
            print(" ".join(seg_tags))

            print("\n=== NER标签对比 ===")
            ner_tags = []
            for c, t, p in zip(text, ner_true, ner_pred):
                ner_t = ner_inv_map.get(t.item(), 'O') if isinstance(t, torch.Tensor) else ner_inv_map.get(t, 'O')
                ner_p = ner_inv_map.get(p.item(), 'O') if isinstance(p, torch.Tensor) else ner_inv_map.get(p, 'O')
                color = 'green' if t == p else 'red'
                ner_tags.append(colored(f"{c}({ner_t}->{ner_p})", color))
            print(" ".join(ner_tags))
            print("="*58 + "\n")


        sample_idx = 0 # Visualize the first sample in the batch
        text_len = np.count_nonzero(seg_labels[sample_idx] != -100)
        seg_true_sample = seg_labels[sample_idx][:text_len]
        seg_pred_sample = seg_pred[sample_idx].argmax(-1)[:text_len]
        ner_true_sample = ner_labels[sample_idx][:text_len]
        ner_pred_sample = ner_pred[sample_idx].argmax(-1)[:text_len]

        visualize_sequence(seg_true_sample, seg_pred_sample, ner_true_sample, ner_pred_sample, seg_label_map, ner_label_map)

        print("=== 详细分类报告 ===")
        print(classification_report(
            cls_labels,
            cls_preds_active,
            target_names=list(cls_label_map.keys()),
            zero_division=0
        ))

    metrics = {
        'accuracy':     avg_acc,
        'seg_accuracy': seg_acc,
        'seg_loss':     seg_loss,
        'seg_f1':       seg_f1,
        'seg_recall':   seg_recall,
        'cls_accuracy': cls_acc,
        'cls_loss':     cls_loss,
        'cls_f1':       cls_f1,
        'cls_recall':   cls_recall,
        'ner_accuracy': ner_acc,
        'ner_loss':     ner_loss,
        'ner_f1':       ner_f1,
        'ner_recall':   ner_recall,
        'total_loss':   total_loss,
    }

    return metrics

def mtl_train( args, tokenizer, device, is_final_eval: bool = False ):
    train_dataset = TextDataset(args.train_data, tokenizer, augment=args.augment)
    test_dataset = TextDataset(args.test_data, tokenizer)

    config = MultiTaskModelConfig(
        model_name=args.pretrained_model,
        attn_implementation="sdpa"
        )
    model = MultiTaskModel(config).to(device)

    # 加载已有模型检查点
    output_dir = f"{args.output_dir}/mtl"
    if args.augment:
        output_dir = f"{output_dir}_aug"
    best_model_path = f"{output_dir}/checkpoint-best"
    if os.path.exists(best_model_path):
        model = MultiTaskModel.from_pretrained(best_model_path, config=config).to(device)
        print("成功加载最佳模型")

    # 初始化参数冻结
    if args.freeze_cls:
        for param in model.cls_fc.parameters( ):
            param.requires_grad = False
        for layer in model.encoder.bert.encoder.layer[:6]:
            for param in layer.parameters( ):
                param.requires_grad = False
        print("已冻结分类层参数")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        fp16=True,
        label_names=['seg_labels', 'cls_labels', 'ner_labels'],
        dataloader_num_workers=4,
        include_inputs_for_metrics=False, #可视化不依赖于原始文本输入
        gradient_accumulation_steps=2
        )

    num_training_steps = len(train_dataset) // args.batch_size * args.epochs \
        if len(train_dataset) % args.batch_size == 0 else (len(train_dataset) // args.batch_size + 1) * args.epochs
    optimizer = AdamW(model.parameters( ), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler_warmup = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * 0.15),
        num_training_steps=num_training_steps
        )

    compute_metrics_fn = partial(mtl_compute_metrics, visualize=is_final_eval)

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        optimizers=(optimizer, scheduler_warmup),
        compute_metrics=compute_metrics_fn,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.patience),
            UnfreezeCallback(freeze=args.freeze_cls, unfreeze_epoch=args.unfreeze_epoch)
            ]
        )

    return trainer

if __name__ == '__main__':
    args = parse_args( )
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast=True)
    device = torch.device('cuda' if torch.cuda.is_available( ) else 'cpu')

    mtl_train(args, tokenizer, device, is_final_eval=True)
