import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFocalLoss(nn.Module):
    def __init__( self, gamma=2.0 ):
        super( ).__init__( )
        self.gamma = gamma

    def forward( self, inputs, targets ):
        ce_loss = F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1), ignore_index=-100)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean( )

class ClsCrossEntropy(nn.Module):
    def forward( self, inputs, targets ):
        return F.cross_entropy(inputs, targets)

class NerFocalLoss(nn.Module):
    def __init__( self, alpha=0.25 ):
        super( ).__init__( )
        self.alpha = alpha

    def forward( self, inputs, targets ):
        ce_loss = F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1), ignore_index=-100)
        return self.alpha * ce_loss.mean( )
