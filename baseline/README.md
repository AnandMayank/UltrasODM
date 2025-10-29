# Baseline Model

## Overview

This directory contains the baseline implementation using EfficientNet-B1 with optical flow integration.

## Key Features

- EfficientNet-B1 backbone for feature extraction
- Optical flow estimation for motion analysis
- Multi-component loss function (MSE, correlation, speed)
- Proven architecture with good baseline performance

## Files

- `train_baseline.py`: Training script
- `network_comparison.py`: Model architecture
- `comparison_losses.py`: Loss function implementations

## Usage

```bash
python baseline/train_baseline.py --config config/baseline_config.yaml
```

## Performance

- Average Point Distance: 0.45mm
- Training Time: ~8 hours
- Parameters: 12.3M

See `docs/RESULTS.md` for detailed performance metrics.
