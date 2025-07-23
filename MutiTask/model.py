import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from transformers import BertModel, PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


# 多任务模型
class SharedEncoder(nn.Module):
    def __init__( self, model_name='bert-base-chinese' ):
        super( ).__init__( )
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

    def forward( self, input_ids, attention_mask ):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state, outputs.pooler_output

class MultiTaskModelConfig(PretrainedConfig):
    def __init__( self, model_name='bert-base-chinese', attn_implementation=None, **kwargs ):
        super( ).__init__(**kwargs)
        self.model_name = model_name
        self.attn_implementation = attn_implementation

class MultiTaskModel(PreTrainedModel):
    def __init__( self, config, seg_labels=3, cls_labels=9, ner_labels=21, dropout=0.1 ):
        super( ).__init__(config)
        self.encoder = SharedEncoder(config.model_name)
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
        self.seg_attention = MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=16,
            batch_first=True,
            dropout=dropout
            )
        self.seg_fc = nn.Linear(hidden_size, seg_labels)

        # 分类任务
        self.cls_attention = MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=16,
            batch_first=True,
            dropout=dropout
            )
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
        self.ner_gate = nn.Linear(hidden_size * 2 + cls_labels, hidden_size)
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
        sequence_output, pooled_output = self.encoder(input_ids, attention_mask)
        sequence_output = self.dropout(sequence_output)

        # 分词任务
        seg_out, _ = self.seg_lstm(sequence_output)
        seg_out = self.seg_bn(seg_out.permute(0, 2, 1)).permute(0, 2, 1)
        seg_attn, _ = self.seg_attention(
            query=seg_out,
            key=seg_out,
            value=seg_out,
            key_padding_mask=(attention_mask == 0)
            )
        seg_logits = self.seg_fc(seg_attn)

        # 分类任务
        cls_attn, _ = self.cls_attention(
            query=sequence_output,
            key=sequence_output,
            value=sequence_output,
            key_padding_mask=(attention_mask == 0)
            )
        cls_pool = self.cls_pool(cls_attn.permute(0, 2, 1)).squeeze( )

        cls_combined = torch.cat([cls_pool, pooled_output], dim=-1)
        cls_gate = self.sigmoid(self.cls_gate(cls_combined))
        cls_input = cls_gate * cls_pool + (1 - cls_gate) * pooled_output
        cls_logits = self.cls_fc(cls_input)

        # 实体识别任务（融合分词信息）
        cls_expanded = cls_logits.unsqueeze(1).expand(-1, sequence_output.size(1), -1)
        ner_combined = torch.cat([sequence_output, seg_out, cls_expanded], dim=-1)
        ner_gate = self.sigmoid(self.ner_gate(ner_combined))  # 动态权重
        ner_input = ner_gate * sequence_output + (1 - ner_gate) * seg_out  # 门控融合

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
        model_name='bert-base-chinese',
        attn_implementation="eager"  # 关键修复：禁用优化注意力机制
        )
    model = MultiTaskModel(config).to(device)
