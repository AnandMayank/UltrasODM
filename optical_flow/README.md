# Optical Flow Module

## Overview

This directory contains the optical flow implementation for motion analysis in ultrasound sequences.

## Key Features

- Lightweight optical flow network (FlowNet)
- Efficient feature extraction from frame pairs
- Adaptive pooling for fixed output size
- Optimized for ultrasound image characteristics

## Architecture

### FlowNet
```
Frame Pair (Concatenated)
    ↓
Conv2D(64) + ReLU
    ↓
Conv2D(128) + ReLU
    ↓
Adaptive AvgPool (4×4)
    ↓
Fully Connected (256)
    ↓
Flow Features
```

## Files

- `optical_flow.py`: FlowNet implementation

## Usage

```python
from optical_flow.optical_flow import FlowNet

# Create flow network
flow_net = FlowNet(in_channels=2)

# Extract flow features
frame1 = torch.randn(B, 1, H, W)
frame2 = torch.randn(B, 1, H, W)
flow_features = flow_net(frame1, frame2)
```

## Integration

This module is integrated into:
- Baseline model (basic optical flow)
- Optical Flow + Mamba model (enhanced multi-scale flow)

## Performance Impact

Using optical flow improves point distance accuracy by ~43% compared to models without motion analysis.

See `docs/RESULTS.md` for ablation study results.
