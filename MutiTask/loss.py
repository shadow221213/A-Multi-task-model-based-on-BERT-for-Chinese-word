import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__( self, alpha=0.25, gamma=2.0, ignore_index=-100 ):
        super( ).__init__( )
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward( self, inputs, targets ):
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean( )

class MultiTaskLoss(nn.Module):
    def __init__( self, num_tasks=3 ):
        """
        Args:
            num_tasks (int): 任务数
        """
        super( ).__init__( )
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.focal_seg = FocalLoss(alpha=0.5, gamma=2.0)
        self.focal_cls = FocalLoss(alpha=0.25, gamma=1.5)
        self.focal_ner = FocalLoss(alpha=0.25, gamma=3.0)

    def forward( self, seg_pred, seg_true, cls_pred, cls_true, ner_pred, ner_true ):
        """
        Args:
            seg_pred (Tensor): [batch_size, seq_len, seg_labels] CRF发射分数
            seg_true (Tensor): [batch_size, seq_len] 真实分词标签
            cls_pred (Tensor): [batch_size, cls_labels] 分类输出
            cls_true (Tensor): [batch_size] 真实分类标签
            ner_pred (Tensor): [batch_size, seq_len, ner_labels] NER输出
            ner_true (Tensor): [batch_size, seq_len] 真实NER标签
        """
        # 分词损失
        seg_loss = self.focal_seg(seg_pred.view(-1, seg_pred.size(-1)), seg_true.view(-1))
        # 分类损失
        cls_loss = self.focal_cls(cls_pred, cls_true)
        # 实体识别损失
        ner_loss = self.focal_ner(ner_pred.view(-1, ner_pred.size(-1)), ner_true.view(-1))

        seg_weight = 1.5
        cls_weight = 0.1
        ner_weight = 1.0
        total_loss = (
                seg_weight * torch.exp(-self.log_vars[0]) * seg_loss + 0.5 * self.log_vars[0] +
                # cls_weight * torch.exp(-self.log_vars[1]) * cls_loss + 0.5 * self.log_vars[1] +
                ner_weight * torch.exp(-self.log_vars[2]) * ner_loss + 0.5 * self.log_vars[2]
        )
        return total_loss.mean( )
