# UltrasODM: A Dual-Stream Optical-Flow–Mamba Network for 3D Freehand Ultrasound Reconstruction

*Accepted at the Proceedings of 2rd AI for Medicine and Healthcare Bridge Program at AAAI 2026*\
>**UltrasODM: A Dual-Stream Optical-Flow–Mamba Network for Trackerless 3D Freehand Ultrasound Reconstruction**\
> Mayank Anand*, Ujair Alam, Surya Prakash, Priya Shukla, G.C Nandi, Dom\`{e}nec Puig\
>*2nd AI Bridge for Medicine & Healthcare, AAAI 2026 (Poster)*

## Overview

This repository contains the implementation of UltrasODM , a deep learning framework for trackerless 3D freehand ultrasound reconstruction. The framework combines video patch embedding, optical flow analysis, and bidirectional Mamba blocks to achieve sub-millimeter accuracy in ultrasound pose estimation.

## Key Features

- **Baseline Model**: EfficientNet-based architecture with optical flow integration
- **Optical Flow Module**: Enhanced motion dynamics extraction using Lucas-Kanade flow estimation
- **Optical Flow + Mamba**: Integration of selective state space models for temporal sequence modeling
- **Dual Mamba Architecture**: Bidirectional Mamba blocks with FPS/NPS sampling for point cloud processing

## Architecture Overview

The framework consists of four main implementations:

### 1. Baseline Model
- EfficientNet-B1 backbone for feature extraction
- Optical flow integration for motion analysis
- Multi-component loss function (MSE, correlation, velocity)

### 2. Optical Flow Enhanced Model
- Enhanced optical flow estimation with multi-scale feature extraction
- Motion magnitude estimation for adaptive feature fusion
- Velocity processor for temporal consistency

### 3. Optical Flow + Mamba Model
- Video patch embedding with adjustable window mechanisms
- Inner Mamba block for initial temporal processing
- FPS/NPS sampling for spatial attention
- Bidirectional Mamba with selective scan algorithm

### 4. Dual Mamba Model
- State space model (SSM) layers with discretization
- Dual-branch processing (FPS and NPS orders)
- Combined feature fusion with restored ordering

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.1.0+
- CUDA 11.8+ (for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/AnandMayank/UltrasODM.git
cd UltrasODM

# Create conda environment
conda create -n ultrasom python=3.9
conda activate ultrasom

# Install dependencies
pip install -r requirements.txt
pip install pytorch3d --no-deps -c pytorch3d
```

## Dataset Structure

The framework expects data in the following format:

```
data/
├── frames_transfs/
│   ├── 000/
│   │   ├── RH_rotation.h5
│   │   └── LH_rotation.h5
│   └── ...
├── landmarks/
│   ├── landmark_000.h5
│   └── ...
└── calib_matrix.csv
```

Each .h5 file contains:
- `frames`: Ultrasound frames (N, H, W)
- `tforms`: Transformation matrices (N, 4, 4)

## Usage

### Training Baseline Model

```bash
python baseline/train_baseline.py --config config/baseline_config.yaml
```

### Training Optical Flow + Mamba Model

```bash
python optical_flow_mamba/train_optical_flow_mamba.py --config config/mamba_config.yaml
```

### Training Dual Mamba Model

```bash
python dual_mamba/train_dual_mamba.py --config config/dual_mamba_config.yaml
```

## Model Architecture Details

### Video Patch Embedding

The video patch embedding module processes video frames into patch embeddings with:
- Adjustable window size for different temporal contexts
- Enhanced temporal encoding with learnable patterns
- Causal sequence modeling for real-time processing

### Optical Flow Integration

The optical flow module extracts motion features through:
- Multi-scale flow feature extraction
- Motion magnitude estimation
- Adaptive fusion based on motion dynamics

### Bidirectional Mamba

The bidirectional Mamba implementation provides:
- True bidirectional processing (forward and backward)
- Selective scan algorithm for efficient sequence modeling
- State space model with discretization

### FPS/NPS Sampling

The combined sampling strategy includes:
- Farthest Point Sampling (FPS) for global coverage
- Nearest Point Sampling (NPS) for local patterns
- Spatial attention mechanism for feature selection

## Loss Functions

The framework implements multiple loss components:

1. **MSE Loss**: Mean squared error for pose prediction
2. **Correlation Loss**: Feature correlation for temporal consistency
3. **Velocity Loss**: Motion velocity regularization
4. **Point Loss**: 3D point distance for clinical accuracy

## Results

Performance metrics on the TUS-REC2025 dataset:

| Model | Point Distance (mm) | Training Time | Parameters |
|-------|-------------------|---------------|------------|
| Baseline | 0.45 | 8h | 12M |
| Optical Flow | 0.32 | 10h | 15M |
| Optical Flow + Mamba | 0.23 | 12h | 18M |
| Dual Mamba | 0.19 | 14h | 22M |

Note: Results are representative and may vary based on training configuration.

## Configuration

Model configurations are stored in `config/`:
- `baseline_config.yaml`: Baseline model settings
- `mamba_config.yaml`: Optical Flow + Mamba settings
- `dual_mamba_config.yaml`: Dual Mamba settings

Key configuration parameters:
- `num_frames`: Number of input frames (default: 4)
- `embed_dim`: Embedding dimension (default: 256)
- `num_fps_points`: FPS sampling points (default: 32)
- `num_nps_points`: NPS sampling points (default: 64)
- `mamba_d_state`: Mamba state dimension (default: 64)

## Code Structure

```
UltrasODM/
├── baseline/                 # Baseline model implementation
│   ├── train_baseline.py
│   └── network_baseline.py
├── optical_flow/            # Optical flow module
│   ├── optical_flow.py
│   └── flow_losses.py
├── optical_flow_mamba/      # Optical Flow + Mamba model
│   ├── train_optical_flow_mamba.py
│   ├── network_mamba.py
│   └── video_patch_embedding.py
├── dual_mamba/              # Dual Mamba model
│   ├── train_dual_mamba.py
│   ├── dual_mamba_block.py
│   └── ssm_layer.py
├── utils/                   # Shared utilities
│   ├── loader.py
│   ├── transform.py
│   ├── metrics.py
│   └── plot_functions.py
├── config/                  # Configuration files
├── data/                    # Dataset directory
└── docs/                    # Documentation

```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{ultrasom2025,
  title={UltrasODM: Ultrasound Object Detection and Motion Estimation},
  author={Anonymous},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

## Acknowledgments

This work is based on research in trackerless 3D freehand ultrasound reconstruction and builds upon advances in state space models and selective scan algorithms.

## License

This code is released for academic research purposes only. Commercial use is prohibited.

## Contact

For questions or issues, please open an issue on GitHub or contact the corresponding author through the conference portal.

---

