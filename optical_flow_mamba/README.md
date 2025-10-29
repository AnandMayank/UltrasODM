# Optical Flow + Mamba Model

## Overview

This directory contains the main implementation combining video patch embedding, optical flow analysis, and bidirectional Mamba blocks.

## Key Features

- Video patch embedding with adjustable windows
- Enhanced optical flow integration with multi-scale features
- Bidirectional Mamba blocks for temporal modeling
- FPS/NPS sampling for spatial attention
- Space-time block with dual processing
- Point-focused loss for clinical accuracy (<0.2mm target)

## Architecture Components

### 1. Video Patch Embedding
- Adjustable window size for temporal context
- Enhanced temporal encoding
- Causal sequence modeling

### 2. Optical Flow Module
- Multi-scale flow extraction
- Motion magnitude estimation
- Adaptive feature fusion

### 3. Bidirectional Mamba
- True bidirectional processing (forward + backward)
- State space model with selective scan
- Efficient sequence modeling

### 4. FPS/NPS Sampling
- Combined sampling strategy
- Global coverage (FPS) + Local patterns (NPS)
- Spatial attention mechanism

## Files

- `train_fps_nps_real_mamba.py`: Main training script
- `network_fps_nps_real_mamba.py`: Model architecture
- `simple_losses.py`: Loss function implementations
- `config.py`: Configuration management
- `fps_nps_sampling.py`: FPS/NPS sampling implementation

## Usage

```bash
python optical_flow_mamba/train_fps_nps_real_mamba.py
```

Or with custom configuration:
```bash
python optical_flow_mamba/train_fps_nps_real_mamba.py --config config/mamba_config.yaml
```

## Performance

- Average Point Distance: 0.23mm (near clinical target)
- Training Time: ~12 hours
- Parameters: 18.4M

## Configuration

Key parameters in `config.py`:
- `NUM_FRAMES`: 4 (input sequence length)
- `EMBED_DIM`: 256 (feature dimension)
- `NUM_FPS_POINTS`: 32
- `NUM_NPS_POINTS`: 64
- `MAMBA_D_STATE`: 64

See `docs/ARCHITECTURE.md` for detailed architecture documentation.
