import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, cardinality=32, stride=1, downsample=None):
        super().__init__()
        mid_channels = out_channels // self.expansion
        
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        
        # Grouped convolution
        self.conv2 = nn.Conv3d(
            mid_channels, mid_channels, kernel_size=3, stride=stride,
            padding=1, groups=cardinality, bias=False
        )
        self.bn2 = nn.BatchNorm3d(mid_channels)
        
        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return F.relu(out)

class AttentionBlock(nn.Module):
    """Self-attention module focusing on speckle-rich regions"""
    def __init__(self, in_channels):
        super().__init__()
        self.att_conv = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att_map = self.sigmoid(self.att_conv(x))
        return x * att_map

class DC2Net(nn.Module):
    def __init__(self, block=Bottleneck3D, layers=[3, 4, 6, 3], num_frames=5, cardinality=32, in_channels=1):
        super().__init__()
        self.in_channels = 64
        self.cardinality = cardinality
        
        # Initial layers
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        
        # Residual stages
        self.layer1 = self._make_layer(block, 256, layers[0])
        self.layer2 = self._make_layer(block, 512, layers[1], stride=(1,2,2))
        self.layer3 = self._make_layer(block, 1024, layers[2], stride=(1,2,2))
        self.layer4 = self._make_layer(block, 2048, layers[3], stride=(1,2,2))
        
        # Attention module
        self.attention = AttentionBlock(2048)
        
        # Global feature extraction
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Multiple parallel output heads - one for each frame
        self.parallel_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2048, 256),
                nn.ReLU(),
                nn.Linear(256, 6)  # 6-DoF output per frame
            ) for _ in range(num_frames)
        ])

        # Global feature head for contrastive learning
        self.feature_head = nn.Sequential(
            nn.Linear(2048 * num_frames, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)  # Feature vector for contrastive loss
        )

        self.num_frames = num_frames

    def _make_layer(self, block, out_channels, blocks, stride=(1,1,1)):
        downsample = None
        if stride != (1,1,1) or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

        layers = [block(self.in_channels, out_channels, self.cardinality, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, self.cardinality))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (B, C, D, H, W) = (B, 1, T, 224, 224)
        batch_size = x.size(0)
        actual_num_frames = x.size(2)  # Get actual number of frames from input

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Attention mechanism
        x = self.attention(x)
        # x shape: (B, 2048, T, H', W')

        # Extract features for each frame separately
        frame_outputs = []
        all_frame_features = []

        # Use the minimum of actual frames and available heads
        num_frames_to_process = min(actual_num_frames, self.num_frames)

        for t in range(num_frames_to_process):
            # Extract features for frame t
            frame_feat = x[:, :, t, :, :]  # (B, 2048, H', W')
            frame_feat = F.adaptive_avg_pool2d(frame_feat, (1, 1))  # (B, 2048, 1, 1)
            frame_feat = frame_feat.view(batch_size, -1)  # (B, 2048)
            all_frame_features.append(frame_feat)

            # Pass through corresponding parallel head
            frame_output = self.parallel_heads[t](frame_feat)  # (B, 6)
            frame_outputs.append(frame_output)

        # If we have fewer frames than heads, pad with zeros or repeat last frame
        while len(frame_outputs) < self.num_frames:
            # Repeat the last frame's output
            if len(frame_outputs) > 0:
                frame_outputs.append(frame_outputs[-1])
                all_frame_features.append(all_frame_features[-1])
            else:
                # If no frames processed, create zero outputs
                zero_output = torch.zeros(batch_size, 6, device=x.device)
                zero_features = torch.zeros(batch_size, 2048, device=x.device)
                frame_outputs.append(zero_output)
                all_frame_features.append(zero_features)

        # Stack frame outputs: (B, T, 6)
        traj_params = torch.stack(frame_outputs, dim=1)

        # Concatenate all frame features for contrastive learning
        concatenated_features = torch.cat(all_frame_features, dim=1)  # (B, 2048*T)
        global_features = self.feature_head(concatenated_features)  # (B, 512)

        return traj_params, global_features

# Utility function to create the model
def create_dc2net(num_frames=7, in_channels=1):
    return DC2Net(block=Bottleneck3D, layers=[3, 4, 6, 3], num_frames=num_frames, in_channels=in_channels)

# Example usage
if __name__ == "__main__":
    model = create_dc2net(num_frames=7, in_channels=1)
    input_tensor = torch.randn(2, 1, 7, 224, 224)  # (B, C, T, H, W)
    traj_params, features = model(input_tensor)

    print("Trajectory parameters shape:", traj_params.shape)  # Expected: (2, 7, 6) - per frame 6-DOF
    print("Feature vector shape:", features.shape)  # Expected: (2, 512) - for contrastive learning
    print("Number of parallel heads:", len(model.parallel_heads))  # Should be 7

    # Test individual frame outputs
    print("\nPer-frame outputs:")
    for i in range(traj_params.shape[1]):
        print(f"Frame {i}: {traj_params[0, i, :].shape} -> 6-DOF parameters")