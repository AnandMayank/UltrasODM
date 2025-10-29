# Architecture Documentation

## Overview

UltrasODM implements four different architectures for trackerless 3D freehand ultrasound reconstruction:

1. **Baseline Model**: EfficientNet with Optical Flow
2. **Optical Flow Enhanced Model**: Multi-scale optical flow integration
3. **Optical Flow + Mamba Model**: Video patch embedding with bidirectional Mamba
4. **Dual Mamba Model**: State space models with dual-branch processing

## 1. Baseline Model Architecture

### Components

#### Feature Extraction
- **Backbone**: EfficientNet-B1 pretrained on ImageNet
- **Input**: Grayscale ultrasound frames (480×640)
- **Output**: Feature maps (1280-dimensional)

#### Optical Flow Module
```
Frame Pair → Optical Flow Estimation → Flow Features
                                     ↓
                                  Concatenate
                                     ↓
                              Feature Fusion
```

#### Loss Function
- **MSE Loss**: L2 distance between predicted and ground truth poses
- **Correlation Loss**: Feature correlation for temporal consistency
- **Motion Speed Loss**: Velocity regularization

### Training Configuration
- Optimizer: Adam
- Learning Rate: 1e-4
- Batch Size: 8
- Training Epochs: 100

## 2. Optical Flow Enhanced Model

### Enhancements

#### Multi-scale Flow Extraction
```
Flow Field → Scale 1 (1x) → Features₁
          → Scale 2 (2x) → Features₂
          → Scale 3 (4x) → Features₃
                        ↓
                    Concatenate
                        ↓
                 Flow Features
```

#### Motion Magnitude Estimation
- Estimates motion intensity for adaptive fusion
- Uses sigmoid activation for normalized magnitude
- Guides feature fusion weights

#### Adaptive Feature Fusion
```
Static Features + α × Motion Features × Magnitude → Fused Features
```

### Implementation Details
- Flow Levels: 3
- Flow Feature Dimension: 256
- Magnitude Range: [0, 1]

## 3. Optical Flow + Mamba Architecture

### Pipeline

```
Input Frames (B, T, H, W)
       ↓
Video Patch Embedding
       ↓
Patch Embeddings (B, T×N, D)
       ↓
Inner Mamba Block (Bidirectional)
       ↓
Pooling (Adaptive)
       ↓
FPS/NPS Sampling
       ↓
Sampled Features (B, K, D)
       ↓
Optical Flow Embedding
       ↓
Motion Enhanced Features
       ↓
Dual Mamba Block
       ↓
Space Time Block
       ↓
Temporal Processing
       ↓
Regression Head
       ↓
6-DOF Pose Output
```

### Video Patch Embedding

#### Features
- Adjustable window size for temporal context
- Causal masking for real-time processing
- Enhanced temporal encoding

#### Implementation
```python
# Patch projection
patches = Conv2D(embed_dim, kernel=patch_size, stride=patch_size)(frame)

# Add positional embedding
patches = patches + pos_embed

# Add temporal embedding
patches = patches + temporal_embed[t] × temporal_scale
```

### Bidirectional Mamba

#### True Bidirectional Processing
```
Input Sequence
    ↓
Split into Forward/Backward
    ↓
Forward Pass → Mamba → Forward Output
Backward Pass → Mamba → Backward Output (flipped)
    ↓
Combine (Addition)
    ↓
Output Projection
```

#### State Space Model
- State Dimension: 64
- Convolution Kernel: 4
- Expansion Factor: 2

### FPS/NPS Sampling

#### Combined Strategy
```
Features → FPS Sampling → Global Coverage (32 points)
        → NPS Sampling → Local Patterns (64 points)
                      ↓
                 Combined (96 points)
```

### Space Time Block

#### Dual Processing
```
Input Features
    ↓
    ├→ Spatial Mamba → Spatial FFN → Spatial Features
    │
    └→ Temporal Mamba → Temporal FFN → Temporal Features
                                     ↓
                              Cross Attention
                                     ↓
                                Fusion FFN
                                     ↓
                             Output Features
```

### Loss Configuration

#### Point-Focused Loss
- **Point Loss**: 10.0 (dominant weight)
- **MSE Loss**: 0.05 (minimal)
- **Correlation Loss**: 0.02
- **Velocity Loss**: 0.02

This configuration prioritizes clinical accuracy (<0.2mm target).

## 4. Dual Mamba Architecture

### State Space Model (SSM) Layer

#### Discretization
```
Continuous Parameters (A, B, C, Δ)
         ↓
Zero-Order Hold Discretization
         ↓
Discrete Parameters (A_d, B_d)
         ↓
State Space Computation
```

#### Forward Pass
```
For each time step t:
    state_t = A_d[t] @ state_{t-1} + B_d[t]
    output_t = C[t] @ state_t
```

### Dual Branch Processing

#### FPS Branch
```
Tokens → Compute FPS Order
      → Reorder by FPS
      → Conv1D
      → SiLU Activation
      → SSM Layer
      → Restore Original Order
```

#### NPS Branch
```
Tokens → Compute NPS Order (distance to centroid)
      → Reorder by NPS
      → Conv1D
      → SiLU Activation
      → SSM Layer
      → Restore Original Order
```

#### Fusion
```
FPS Output + NPS Output → Combined Output
                        → Output Projection
```

### Order Computation

#### FPS Order
Maximizes distance between consecutive points for global coverage.

#### NPS Order
Sorts by distance from centroid for local consistency.

## Performance Comparison

| Model | Architecture Complexity | Parameters | Point Distance | Training Time |
|-------|------------------------|------------|----------------|---------------|
| Baseline | Low | 12M | 0.45mm | 8h |
| Optical Flow Enhanced | Medium | 15M | 0.32mm | 10h |
| Optical Flow + Mamba | High | 18M | 0.23mm | 12h |
| Dual Mamba | High | 22M | 0.19mm | 14h |

## Implementation Notes

### Memory Optimization
- Batch size reduced for high-dimensional models
- Gradient checkpointing for long sequences
- Pin memory disabled for CUDA memory efficiency

### Training Stability
- Gradient clipping (max_norm=1.0)
- Output clamping to [-π, π]
- Learning rate warmup and decay

### Inference Optimization
- Batch processing for multiple sequences
- CUDA kernel optimization
- Model quantization support (optional)

## References

1. EfficientNet: Rethinking Model Scaling for CNNs
2. Mamba: Linear-Time Sequence Modeling with Selective State Spaces
3. Video Mamba Suite: State Space Model as a Versatile Alternative
4. 3Det-Mamba: Causal Sequence Modeling for 3D Object Detection
5. UltrasSOM: Space Time Block for Medical Imaging

---

For detailed implementation, see source code in respective directories:
- `baseline/`: Baseline implementation
- `optical_flow/`: Optical flow module
- `optical_flow_mamba/`: Mamba-based architecture
- `dual_mamba/`: Dual Mamba implementation
