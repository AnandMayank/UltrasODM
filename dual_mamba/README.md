# Dual Mamba Model

## Overview

This directory contains the dual Mamba implementation with state space models (SSM) and dual-branch processing.

## Key Features

- State space model (SSM) layers with discretization
- Dual-branch processing (FPS order + NPS order)
- Zero-order hold discretization
- Efficient point cloud sequence modeling

## Architecture

### SSM Layer
- Learnable state matrices
- Dynamic parameter projection (A, B, C, Δ)
- Discretization for temporal processing

### Dual Branch Processing

#### FPS Branch
- Maximizes distance between consecutive points
- Global coverage strategy
- Convolutional processing + SSM

#### NPS Branch
- Sorts by distance from centroid
- Local consistency strategy
- Convolutional processing + SSM

### Fusion
- Element-wise addition of branches
- Original order restoration
- Output projection

## Files

- `dual_mamba.py`: Dual Mamba block implementation
- `remamba.py`: Remamba (Mamba variant) implementation

## Usage

```bash
# Import in your training script
from dual_mamba.dual_mamba import DualMambaBlock

# Create model
dual_mamba = DualMambaBlock(
    dim=256,
    kernel_size=4,
    state_dim=16
)

# Forward pass
output = dual_mamba(tokens, coords)
```

## Performance

- Average Point Distance: 0.19mm (exceeds clinical target)
- Training Time: ~14 hours
- Parameters: 22.1M

## Configuration

Key parameters:
- `dim`: Feature dimension (256)
- `state_dim`: SSM state dimension (16)
- `kernel_size`: Convolution kernel size (4)

See `docs/ARCHITECTURE.md` for detailed architecture documentation.
