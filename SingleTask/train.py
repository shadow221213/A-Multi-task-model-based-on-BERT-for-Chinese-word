import os

import torch
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from transformers import (AutoTokenizer, EarlyStoppingCallback, EvalPrediction,
                          Trainer, TrainerCallback, TrainingArguments,
                          get_linear_schedule_with_warmup)

from data.preprocess import TextDataset
from SingleTask.loss import ClsCrossEntropy, NerFocalLoss, SegFocalLoss
from SingleTask.model import (ClsModel, ClsModelConfig,
                   NerModel, NerModelConfig, SegModel,
                   SegModelConfig)
from utils.tools import parse_args


def collate_fn( batch ):
    return {
        'input_ids':      torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'seg_labels':     torch.stack([x['seg_labels'] for x in batch]),
        'cls_labels':     torch.stack([x['cls_labels'] for x in batch]),
        'ner_labels':     torch.stack([x['ner_labels'] for x in batch])
        }

class SingleTaskTrainer(Trainer):
    def __init__( self, task_type, *args, **kwargs ):
        super( ).__init__(*args, **kwargs)
        self.task_type = task_type
        self.best_metric = -1

    def compute_loss( self, model, inputs, return_outputs=False ):
        labels = inputs[f"{self.task_type}_labels"]
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        logits = outputs["logits"]

        if self.task_type == "seg":
            loss_fct = SegFocalLoss( )
            loss = loss_fct(logits, labels)
        elif self.task_type == "cls":
            loss_fct = ClsCrossEntropy( )
            loss = loss_fct(logits, labels)
        elif self.task_type == "ner":
            loss_fct = NerFocalLoss( )
            loss = loss_fct(logits, labels)

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

def stl_compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    # 过滤掉填充标签 -100
    mask = labels != -100
    active_preds = predictions.argmax(-1)[mask]
    active_labels = labels[mask]

    acc = accuracy_score(active_labels, active_preds)

    metrics = {
        'accuracy': acc,
    }

    return metrics

def stl_train(task_type, args, tokenizer, device):
    train_dataset = TextDataset(args.train_data, tokenizer, augment=args.augment)
    test_dataset = TextDataset(args.test_data, tokenizer)

    # 加载模型
    output_dir = f"{args.output_dir}/{task_type}"
    best_model_path = f"{output_dir}/checkpoint-best"

    if task_type == "seg":
        config = SegModelConfig(model_name=args.pretrained_model)
        model = SegModel(config).to(device)
        if os.path.exists(best_model_path):
            model = SegModel.from_pretrained(best_model_path, config=config).to(device)
            print("成功加载最佳模型")
    elif task_type == "cls":
        config = ClsModelConfig(model_name=args.pretrained_model)
        model = ClsModel(config).to(device)
        if os.path.exists(best_model_path):
            model = ClsModel.from_pretrained(best_model_path, config=config).to(device)
            print("成功加载最佳模型")
    elif task_type == "ner":
        config = NerModelConfig(model_name=args.pretrained_model)
        model = NerModel(config).to(device)
        if os.path.exists(best_model_path):
            model = NerModel.from_pretrained(best_model_path, config=config).to(device)
            print("成功加载最佳模型")
    else:
        raise ValueError('must be in ["seg", "cls", "ner"]')

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
        label_names=[f'{task_type}_labels'],
        dataloader_num_workers=4,
        include_inputs_for_metrics=False,
        gradient_accumulation_steps=2
    )

    # 优化器
    num_training_steps = len(train_dataset) // args.batch_size * args.epochs \
        if len(train_dataset) % args.batch_size == 0 else (len(train_dataset) // args.batch_size + 1) * args.epochs
    optimizer = AdamW(model.parameters( ), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler_warmup = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * 0.15),
        num_training_steps=num_training_steps
        )

    # 训练器
    trainer = SingleTaskTrainer(
        task_type=task_type,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        optimizers=(optimizer, scheduler_warmup),
        compute_metrics=stl_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    return trainer
