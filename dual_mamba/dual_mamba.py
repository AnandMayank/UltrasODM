import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SSMLayer(nn.Module):
    """State Space Model (SSM) Layer with discretization"""
    def __init__(self, dim, state_dim=16):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        
        # Projection layers for SSM parameters
        self.A_proj = nn.Linear(dim, state_dim)
        self.B_proj = nn.Linear(dim, state_dim)
        self.C_proj = nn.Linear(dim, state_dim)
        self.D_proj = nn.Linear(dim, state_dim)  # Δ (delta) parameter
        
        # Output projection
        self.out_proj = nn.Linear(state_dim, dim)
        
        # Learnable state matrix (initialized as diagonal)
        self.A = nn.Parameter(torch.eye(state_dim).float())

    def discretize(self, A, B, delta):
        """Discretize continuous parameters using zero-order hold"""
        delta_A = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, state_dim, state_dim)
        delta_B = (delta_A - torch.eye(self.state_dim).to(A.device)) @ torch.linalg.inv(A)
        delta_B = delta_B @ (delta.unsqueeze(-1) * B.unsqueeze(-1))  # (B, L, state_dim)
        return delta_A, delta_B

    def forward(self, x):
        """Input shape: (batch, seq_len, dim)"""
        batch_size, seq_len, _ = x.shape
        
        # Project to get SSM parameters
        A = self.A_proj(x)  # (B, L, state_dim)
        B = self.B_proj(x)  # (B, L, state_dim)
        C = self.C_proj(x)  # (B, L, state_dim)
        delta = F.softplus(self.D_proj(x))  # (B, L, state_dim)
        
        # Discretize parameters
        A_disc, B_disc = self.discretize(self.A, B, delta)
        
        # State space computation (simplified)
        states = torch.zeros(batch_size, self.state_dim).to(x.device)
        outputs = []
        
        for t in range(seq_len):
            states = torch.einsum('bls,bns->bn', A_disc[:, t], states) + B_disc[:, t]
            out_t = torch.einsum('bls,bs->bl', C[:, t].unsqueeze(1), states)
            outputs.append(out_t)
        
        y = torch.stack(outputs, dim=1)  # (B, L, dim)
        return self.out_proj(y)

class DualMambaBlock(nn.Module):
    def __init__(self, dim, kernel_size=4, state_dim=16):
        """
        Dual Mamba Block for 3D point clouds
        
        Args:
            dim: Input feature dimension
            kernel_size: Convolution kernel size
            state_dim: SSM state dimension
        """
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        
        # Shared normalization layer
        self.norm = nn.LayerNorm(dim)
        
        # Projection layers for each branch
        self.proj_fps = nn.Linear(dim, dim)
        self.proj_nps = nn.Linear(dim, dim)
        
        # Convolution layers
        self.conv_fps = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2)
        self.conv_nps = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2)
        
        # Activation
        self.act = nn.SiLU()
        
        # SSM layers
        self.ssm_fps = SSMLayer(dim, state_dim)
        self.ssm_nps = SSMLayer(dim, state_dim)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, tokens, coords):
        """
        Args:
            tokens: Patch tokens (B, K, C)
            coords: Keypoint coordinates (B, K, 3)
        Returns:
            output: (B, K, C)
        """
        B, K, C = tokens.shape
        
        # 1. Compute orders
        with torch.no_grad():
            # FPS order - Maximize distance between consecutive points
            fps_order = self.compute_fps_order(coords)
            
            # NPS order - Sort by distance from centroid (maintains local consistency)
            centroid = torch.mean(coords, dim=1, keepdim=True)
            dist_to_centroid = torch.norm(coords - centroid, dim=-1)
            nps_order = torch.argsort(dist_to_centroid, dim=1)
        
        # 2. Reorder tokens
        tokens_fps = torch.gather(tokens, 1, fps_order.unsqueeze(-1).expand(-1, -1, C))
        tokens_nps = torch.gather(tokens, 1, nps_order.unsqueeze(-1).expand(-1, -1, C))
        
        # 3. Process FPS branch
        x_fps = self.norm(tokens_fps)
        x_fps = self.proj_fps(x_fps)
        x_fps = rearrange(x_fps, 'b k c -> b c k')
        x_fps = self.conv_fps(x_fps)
        x_fps = rearrange(x_fps, 'b c k -> b k c')
        x_fps = self.act(x_fps)
        x_fps = self.ssm_fps(x_fps)
        
        # 4. Process NPS branch
        x_nps = self.norm(tokens_nps)
        x_nps = self.proj_nps(x_nps)
        x_nps = rearrange(x_nps, 'b k c -> b c k')
        x_nps = self.conv_nps(x_nps)
        x_nps = rearrange(x_nps, 'b c k -> b k c')
        x_nps = self.act(x_nps)
        x_nps = self.ssm_nps(x_nps)
        
        # 5. Combine branches
        combined = x_fps + x_nps
        
        # 6. Restore original order (inverse permutation)
        # For FPS branch
        _, fps_inv = torch.sort(fps_order, dim=1)
        x_fps_restored = torch.gather(x_fps, 1, fps_inv.unsqueeze(-1).expand(-1, -1, C))
        
        # For NPS branch
        _, nps_inv = torch.sort(nps_order, dim=1)
        x_nps_restored = torch.gather(x_nps, 1, nps_inv.unsqueeze(-1).expand(-1, -1, C))
        
        # Combine restored outputs
        combined_restored = x_fps_restored + x_nps_restored
        
        # Final projection
        output = self.out_proj(combined_restored)
        
        return output

    def compute_fps_order(self, coords, start_idx=0):
        """Compute Farthest Point Sampling (FPS) order for each sample in batch"""
        B, K, _ = coords.shape
        device = coords.device
        
        # Initialize order tensor
        order = torch.zeros(B, K, dtype=torch.long, device=device)
        remaining_mask = torch.ones(B, K, dtype=torch.bool, device=device)
        
        # Start with random point
        order[:, 0] = start_idx
        remaining_mask[torch.arange(B), start_idx] = False
        
        # Compute pairwise distances
        dists = torch.cdist(coords, coords, p=2)  # (B, K, K)
        
        for i in range(1, K):
            # Get last selected point
            last_pts = order[:, i-1]
            
            # Get distances to last selected point
            last_dists = dists[torch.arange(B), last_pts]  # (B, K)
            
            # Find farthest point from current selection
            # Set selected points to -inf so they're not chosen
            last_dists[~remaining_mask] = -float('inf')
            farthest_idxs = torch.argmax(last_dists, dim=1)  # (B,)
            
            # Update order and mask
            order[:, i] = farthest_idxs
            remaining_mask[torch.arange(B), farthest_idxs] = False
        
        return order