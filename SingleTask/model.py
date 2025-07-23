import torch
import torch.nn as nn
from transformers import BertModel, PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


# 分词任务模型
class SegModelConfig(PretrainedConfig):
    def __init__( self, model_name='bert-base-chinese', **kwargs ):
        super( ).__init__(**kwargs)
        self.model_name = model_name

class SegModel(PreTrainedModel):
    def __init__( self, config, seg_labels=3, dropout=0.1 ):
        super( ).__init__(config)
        self.bert = BertModel.from_pretrained(config.model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        self.seg_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            bidirectional=True,
            batch_first=True
            )
        self.seg_bn = nn.BatchNorm1d(hidden_size)
        self.seg_fc = nn.Linear(hidden_size, seg_labels)

    def forward( self, input_ids, attention_mask ):
        outputs = self.bert(input_ids, attention_mask)
        sequence_output = outputs.last_hidden_state

        seg_out, _ = self.seg_lstm(sequence_output)
        seg_out = self.seg_bn(seg_out.permute(0, 2, 1)).permute(0, 2, 1)
        seg_logits = self.seg_fc(seg_out)

        return {
            "logits": seg_logits
            }

# 分类任务模型
class ClsModelConfig(PretrainedConfig):
    def __init__( self, model_name='bert-base-chinese', **kwargs ):
        super( ).__init__(**kwargs)
        self.model_name = model_name

class ClsModel(PreTrainedModel):
    def __init__( self, config, cls_labels=9, dropout=0.1):
        super( ).__init__(config)
        self.bert = BertModel.from_pretrained(config.model_name)
        hidden_size = self.bert.config.hidden_size

        self.cls_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU( ),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, cls_labels)
            )

    def forward( self, input_ids, attention_mask ):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs.pooler_output

        cls_logits = self.cls_fc(pooled_output)

        return {
            "logits": cls_logits
            }

# NER任务模型
class NerModelConfig(PretrainedConfig):
    def __init__( self, model_name='bert-base-chinese', **kwargs ):
        super( ).__init__(**kwargs)
        self.model_name = model_name

class NerModel(PreTrainedModel):
    def __init__( self, config, ner_labels=21, dropout=0.1):
        super( ).__init__(config)
        self.bert = BertModel.from_pretrained(config.model_name)
        hidden_size = self.bert.config.hidden_size

        self.ner_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            bidirectional=True,
            batch_first=True
            )
        self.ner_bn = nn.BatchNorm1d(hidden_size)
        self.ner_fc = nn.Linear(hidden_size, ner_labels)

    def forward( self, input_ids, attention_mask ):
        outputs = self.bert(input_ids, attention_mask)
        sequence_output = outputs.last_hidden_state

        ner_out, _ = self.ner_lstm(sequence_output)
        ner_out = self.ner_bn(ner_out.permute(0, 2, 1)).permute(0, 2, 1)
        ner_logits = self.ner_fc(ner_out)

        return {
            "logits": ner_logits
            }
