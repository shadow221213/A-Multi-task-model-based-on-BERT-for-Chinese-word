import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from transformers import BertModel, PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

# 多任务模型
class SharedEncoder(nn.Module):
    def __init__( self, model_name='bert-base-chinese', N_shared=8 ):
        super( ).__init__( )
        self.bert = BertModel.from_pretrained(model_name)

        from copy import deepcopy
        self.embeddings = deepcopy(self.bert.embeddings)
        self.shared_layers = nn.ModuleList(deepcopy(layer) for layer in self.bert.encoder.layer[:N_shared])
        self.N_shared = N_shared

        # 为每个任务独立声明后 12-N_shared 层, 这里用 deepcopy 保证权重初始化为同样的预训练权重，但后续可独立更新
        self.seg_layers = nn.ModuleList(deepcopy(self.bert.encoder.layer[N_shared:]))
        self.cls_layers = nn.ModuleList(deepcopy(self.bert.encoder.layer[N_shared:]))
        self.ner_layers = nn.ModuleList(deepcopy(self.bert.encoder.layer[N_shared:]))

        self.hidden_size = self.bert.config.hidden_size

    def _forward_layers( self, layers, hidden_states, attention_mask ):
        """依次过给定的 transformer layer 列表"""
        for layer in layers:
            hidden_states = layer(hidden_states, attention_mask)[0]
        return hidden_states
    def forward( self, input_ids, attention_mask, task='seg' ):
        """
        task: 'seg' | 'cls' | 'ner'，决定走哪一组独立高层
        """
        # 1. embedding & 共享层
        extended_mask = attention_mask[:, None, None, :]  # (B,1,1,L)
        extended_mask = (1.0 - extended_mask) * torch.finfo(torch.float16).min

        hidden_states = self.embeddings(input_ids)
        hidden_states = self._forward_layers(self.shared_layers, hidden_states, extended_mask)

        # 2. 任务专属高层
        if task == 'seg':
            hidden_states = self._forward_layers(self.seg_layers, hidden_states, extended_mask)
        elif task == 'cls':
            hidden_states = self._forward_layers(self.cls_layers, hidden_states, extended_mask)
        elif task == 'ner':
            hidden_states = self._forward_layers(self.ner_layers, hidden_states, extended_mask)
        else:
            raise ValueError("task must be one of {'seg', 'cls', 'ner'}")

        # 3. pooler（复用 BERT 原 pooler）
        pooled = hidden_states[:, 0]  # 简单取 [CLS]，也可接 pooler
        return hidden_states, pooled

class MultiTaskModelConfig(PretrainedConfig):
    def __init__( self, model_name='bert-base-chinese', **kwargs ):
        super( ).__init__(**kwargs)
        self.model_name = model_name

class MultiTaskModel(PreTrainedModel):
    def __init__( self, config, seg_labels=3, cls_labels=9, ner_labels=21, dropout=0.1 ):
        super( ).__init__(config)
        self.encoder = SharedEncoder(config.model_name, N_shared=8)
        hidden_size = self.encoder.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid( )

        # 分词任务（LSTM+注意力机制）
        self.seg_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            bidirectional=True,
            batch_first=True
            )
        self.seg_bn = nn.BatchNorm1d(hidden_size)
        self.seg_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.seg_fc = nn.Linear(hidden_size, seg_labels)

        # 分类任务
        self.cls_pool = nn.AdaptiveAvgPool1d(1)
        self.cls_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.cls_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU( ),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, cls_labels)
            )

        # 实体识别任务（BiLSTM）
        self.ner_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.ner_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            bidirectional=True,
            batch_first=True
            )
        self.ner_bn = nn.BatchNorm1d(hidden_size)
        self.ner_fc = nn.Linear(hidden_size, ner_labels)

    def forward( self, input_ids, attention_mask, **kwargs ):
        # 共享编码
        seq_seg, pool_seg = self.encoder(input_ids, attention_mask, task='seg')
        seq_cls, pool_cls = self.encoder(input_ids, attention_mask, task='cls')
        seq_ner, pool_ner = self.encoder(input_ids, attention_mask, task='ner')

        seq_seg = self.dropout(seq_seg)
        seq_cls = self.dropout(seq_cls)
        seq_ner = self.dropout(seq_ner)

        # 分词任务
        seg_out, _ = self.seg_lstm(seq_seg)
        seg_out = self.seg_bn(seg_out.permute(0, 2, 1)).permute(0, 2, 1)
        seg_logits = self.seg_fc(seg_out)

        # 分类任务
        cls_pool = self.cls_pool(seq_cls.permute(0, 2, 1)).squeeze(-1)
        cls_combined = torch.cat([cls_pool, pool_cls], dim=-1)
        cls_gate = self.sigmoid(self.cls_gate(cls_combined))
        cls_input = cls_gate * cls_pool + (1 - cls_gate) * pool_cls
        cls_logits = self.cls_fc(cls_input)

        # 实体识别任务（融合分词信息）
        ner_combined = torch.cat([seq_ner, seg_out], dim=-1)
        ner_gate = self.sigmoid(self.ner_gate(ner_combined))  # 动态权重
        ner_input = ner_gate * seq_ner + (1 - ner_gate) * seg_out  # 门控融合

        ner_out, _ = self.ner_lstm(ner_input)
        ner_out = self.ner_bn(ner_out.permute(0, 2, 1)).permute(0, 2, 1)
        ner_logits = self.ner_fc(ner_out)

        return {
            "seg_logits": seg_logits,
            "cls_logits": cls_logits,
            "ner_logits": ner_logits
            }

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available( ) else 'cpu')
    config = MultiTaskModelConfig(
        model_name='bert-base-chinese'
        )
    model = MultiTaskModel(config).to(device)
