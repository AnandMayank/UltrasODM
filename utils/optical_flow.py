# utils/optical_flow.py
import torch
import torch.nn as nn

class FlowNet(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Fixed output size regardless of input
        )
        self.fc = nn.Linear(128*4*4, 256)  # Now fixed: 128*4*4 = 2048

    def forward(self, frame1, frame2):
        flow_input = torch.cat([frame1, frame2], dim=1)
        features = self.conv(flow_input)
        return self.fc(features.flatten(1))