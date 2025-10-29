# metrics
import torch
import torch.nn as nn

class PointDistance:
    # calculate the distance between two sets of points
    def __init__(self,paired=True,):
        
        self.paired = paired
    
    def __call__(self,preds,labels):
        if self.paired:
            return ((preds-labels)**2).sum(dim=2).sqrt().mean(dim=(0,2))
        else:
            return ((preds-labels)**2).sum(dim=2).sqrt().mean()
        
# loss.py
class ContrastiveLoss(nn.Module):
    """Triplet loss with hard negative mining"""
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
        self.distance = nn.PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative):
        pos_dist = self.distance(anchor, positive)
        neg_dist = self.distance(anchor, negative)
        losses = torch.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()

    def __call__(self, anchor, positive, negative):
        return self.forward(anchor, positive, negative)
        
