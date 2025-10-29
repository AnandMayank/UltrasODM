import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
import numpy as np
from torchvision.models import resnet18

class ContrastiveFrameGrouping:
    def __init__(self, margin=0.2, delta=5, tau_sim=0.7):
        """
        Args:
            margin (float): Margin α for triplet loss (default: 0.2)
            delta (int): Frame distance for negative sampling (default: 5)
            tau_sim (float): Similarity threshold for DBSCAN clustering
        """
        self.margin = margin
        self.delta = delta
        self.tau_sim = tau_sim
        
        # Initialize ResNet-18 backbone
        self.embedding_network = resnet18(pretrained=True)
        # Modify final layer for embedding
        self.embedding_network.fc = nn.Linear(512, 128)  # 128-dim embeddings
        
    def compute_embeddings(self, frames):
        """Compute embeddings for frames using ResNet backbone"""
        return self.embedding_network(frames)
    
    def triplet_loss(self, anchor, positive, negative):
        """Compute triplet loss with margin"""
        dist_pos = torch.norm(anchor - positive, dim=1) ** 2
        dist_neg = torch.norm(anchor - negative, dim=1) ** 2
        loss = torch.max(torch.zeros_like(dist_pos), 
                        dist_pos - dist_neg + self.margin)
        return loss.mean()
    
    def train_step(self, frames):
        """Single training iteration as per algorithm"""
        batch_size = frames.shape[0]
        
        # Sample anchor, positive and negative
        anchor_idx = np.random.randint(0, batch_size)
        
        # Find valid positive indices (|a-p| ≤ δ)
        valid_pos = [i for i in range(batch_size) 
                    if 0 < abs(i-anchor_idx) <= self.delta]
        pos_idx = np.random.choice(valid_pos)
        
        # Find valid negative indices (|a-n| ≥ Δ)
        valid_neg = [i for i in range(batch_size) 
                    if abs(i-anchor_idx) >= self.delta]
        neg_idx = np.random.choice(valid_neg)
        
        # Compute embeddings
        embeddings = self.compute_embeddings(frames)
        
        # Get triplet embeddings
        anchor_emb = embeddings[anchor_idx]
        pos_emb = embeddings[pos_idx]
        neg_emb = embeddings[neg_idx]
        
        # Compute loss
        loss = self.triplet_loss(anchor_emb, pos_emb, neg_emb)
        
        return loss
    
    def inference(self, frames):
        """Inference stage: compute embeddings and cluster"""
        # Compute embeddings for all frames
        with torch.no_grad():
            embeddings = self.compute_embeddings(frames)
        
        # Convert to numpy for DBSCAN
        embeddings_np = embeddings.cpu().numpy()
        
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=self.tau_sim, min_samples=2)
        labels = clustering.fit_predict(embeddings_np)
        
        # Group frames by cluster
        groups = {}
        for i, label in enumerate(labels):
            if label != -1:  # Ignore noise points
                if label not in groups:
                    groups[label] = []
                groups[label].append(i)
                
        return groups