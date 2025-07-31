import os
from functools import partial

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             recall_score)
from termcolor import colored
from torch.optim import AdamW
from transformers import (AutoTokenizer, EarlyStoppingCallback, EvalPrediction,
                          Trainer, TrainerCallback, TrainingArguments,
                          get_linear_schedule_with_warmup)

from data.preprocess import TextDataset
from MutiTask.loss import MultiTaskLoss
from MutiTask.model import MultiTaskModel, MultiTaskModelConfig
from utils.tools import parse_args


def collate_fn( batch ):
    return {
        'input_ids':      torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'seg_labels':     torch.stack([x['seg_labels'] for x in batch]),
        'cls_labels':     torch.stack([x['cls_labels'] for x in batch]),
        'ner_labels':     torch.stack([x['ner_labels'] for x in batch])
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

def mtl_compute_metrics( eval_pred: EvalPrediction, visualize: bool = False ):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    seg_pred, cls_pred, ner_pred = predictions
    seg_labels, cls_labels, ner_labels = labels

    def masked_accuracy( logits, labels, ignore_index=-100 ):
        # 过滤掉填充标签
        mask = (labels != ignore_index).flatten( )
        preds = logits.argmax(-1).flatten( )[mask]
        targets = labels.flatten( )[mask]
        return accuracy_score(targets, preds)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # 计算各任务指标
    seg_acc = masked_accuracy(seg_pred, seg_labels)
    cls_acc = accuracy_score(cls_labels, cls_pred.argmax(-1))
    ner_acc = masked_accuracy(ner_pred, ner_labels)
    avg_acc = (seg_acc + cls_acc + ner_acc) / 3

    seg_pred = torch.tensor(seg_pred) if not isinstance(seg_pred, torch.Tensor) else seg_pred
    cls_pred = torch.tensor(cls_pred) if not isinstance(cls_pred, torch.Tensor) else cls_pred
    ner_pred = torch.tensor(ner_pred) if not isinstance(ner_pred, torch.Tensor) else ner_pred

    seg_labels = torch.tensor(seg_labels) if not isinstance(seg_labels, torch.Tensor) else seg_labels
    cls_labels = torch.tensor(cls_labels) if not isinstance(cls_labels, torch.Tensor) else cls_labels
    ner_labels = torch.tensor(ner_labels) if not isinstance(ner_labels, torch.Tensor) else ner_labels

    seg_pred_reshaped = seg_pred.view(-1, seg_pred.shape[-1])
    seg_labels_reshaped = seg_labels.view(-1)

    cls_pred_reshaped = cls_pred.view(-1, cls_pred.shape[-1]) if len(cls_pred.shape) > 1 else cls_pred
    cls_labels_reshaped = cls_labels.view(-1)

    ner_pred_reshaped = ner_pred.view(-1, ner_pred.shape[-1]) if len(ner_pred.shape) > 1 else ner_pred
    ner_labels_reshaped = ner_labels.view(-1)

    # 计算各任务的损失
    seg_loss = loss_fn(seg_pred_reshaped, seg_labels_reshaped)  # 计算损失
    cls_loss = loss_fn(cls_pred_reshaped, cls_labels_reshaped).mean( ).item( )  # 分类任务损失
    ner_loss = loss_fn(ner_pred_reshaped, ner_labels_reshaped)  # NER任务损失

    if seg_loss.ndimension( ) > 0:
        seg_loss = seg_loss[ner_labels_reshaped != -100].mean( ).item( )
    else:
        seg_loss = seg_loss.item( )

    if ner_loss.ndimension( ) > 0:
        ner_loss = ner_loss[ner_labels_reshaped != -100].mean( ).item( )
    else:
        ner_loss = ner_loss.item( )

    total_loss = seg_loss + cls_loss + ner_loss

    def get_valid_labels( logits, labels, task_name ):
        preds = np.argmax(logits, axis=-1).flatten( )
        labels = labels.flatten( )
        mask = labels != -100
        return preds[mask], labels[mask]

    # 计算各任务指标
    seg_p, seg_t = get_valid_labels(seg_pred, seg_labels, "seg")
    ner_p, ner_t = get_valid_labels(ner_pred, ner_labels, "ner")

    seg_f1 = f1_score(seg_t, seg_p, average='macro')
    seg_recall = recall_score(seg_t, seg_p, average='macro')
    cls_f1 = f1_score(cls_labels, cls_pred.argmax(-1), average='macro')
    cls_recall = recall_score(cls_labels, cls_pred.argmax(-1), average='macro')
    ner_f1 = f1_score(ner_t, ner_p, average='macro')
    ner_recall = recall_score(ner_t, ner_p, average='macro')

    if visualize:
        seg_label_map = {
            'B': 0,
            'I': 1,
            'O': 2
            }
        cls_label_map = {
            '体育': 0,
            '娱乐': 1,
            '彩票': 2,
            '房产': 3,
            '教育': 4,
            '时政': 5,
            '游戏': 6,
            '社会': 7,
            '财经': 8
            }
        ner_label_map = {
            'O':              0,
            'B-name':         1,
            'I-name':         2,
            'B-organization': 3,
            'I-organization': 4,
            'B-address':      5,
            'I-address':      6,
            'B-government':   7,
            'I-government':   8,
            'B-scene':        9,
            'I-scene':        10,
            'B-game':         11,
            'I-game':         12,
            'B-position':     13,
            'I-position':     14,
            'B-book':         15,
            'I-book':         16,
            'B-company':      17,
            'I-company':      18,
            'B-movie':        19,
            'I-movie':        20,
            }

        def visualize_sequence( seg_true, seg_pred, ner_true, ner_pred, seg_label_map, ner_label_map ):
            text = "这也是张路老师在新浪的足彩推荐中第四次命中头奖！"
            seg_inv_map = {v: k for k, v in seg_label_map.items( )}
            ner_inv_map = {v: k for k, v in ner_label_map.items( )}

            print("\n====" + " 预测结果可视化 " + "====")
            print("=== 原始文本 ===")
            print(text)
            print("\n=== 分词标签对比 ===")
            seg_tags = []
            for c, t, p in zip(text, seg_true, seg_pred):
                seg_t = seg_inv_map.get(t.item( ), f'UNK_{t.item( )}')
                seg_p = seg_inv_map.get(p.item( ), f'UNK_{p.item( )}')
                color = 'green' if t == p else 'red'
                seg_tags.append(colored(f"{c}({seg_t}->{seg_p})", color))
            print(" ".join(seg_tags))

            print("\n=== NER标签对比 ===")
            ner_tags = []
            for c, t, p in zip(text, ner_true, ner_pred):
                ner_t = ner_inv_map.get(t.item( ), f'UNK_{t.item( )}')
                ner_p = ner_inv_map.get(p.item( ), f'UNK_{p.item( )}')
                color = 'green' if t == p else 'red'
                ner_tags.append(colored(f"{c}({ner_t}->{ner_p})", color))
            print(" ".join(ner_tags))
            print("=" * 50 + "\n")

        sample_idx = 155  # Visualize the first sample in the batch
        seg_true = seg_labels[sample_idx][seg_labels[sample_idx] != -100]
        seg_pr = seg_pred[sample_idx].argmax(-1)[seg_labels[sample_idx] != -100]
        ner_true = ner_labels[sample_idx][ner_labels[sample_idx] != -100]
        ner_pr = ner_pred[sample_idx].argmax(-1)[ner_labels[sample_idx] != -100]

        visualize_sequence(seg_true, seg_pr, ner_true, ner_pr, seg_label_map, ner_label_map)

        print("=== 详细分类报告 ===")
        print(classification_report(
            cls_labels,
            cls_pred.argmax(-1),
            target_names=list(cls_label_map.keys( )),
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

    config = MultiTaskModelConfig(model_name=args.pretrained_model)
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
        per_device_train_batch_size=args.batch_size // args.grad_accumulation_steps,
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
        dataloader_pin_memory=True,
        label_names=['seg_labels', 'cls_labels', 'ner_labels'],
        include_inputs_for_metrics=False,  # 可视化不依赖于原始文本输入
        gradient_accumulation_steps=args.grad_accumulation_steps
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
