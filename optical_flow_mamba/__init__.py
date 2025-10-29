"""
Simplified FPS/NPS Package for Ultrasound Pose Regression

A clean, focused implementation of FPS/NPS sampling for ultrasound pose regression,
designed based on the analysis that simpler approaches often outperform complex 
multi-loss architectures.

Key Features:
- EfficientNet-B1 backbone (proven effective in TUS-REC baseline)
- FPS/NPS sampling for temporal modeling
- PointDistance loss only (most effective)
- ~10-15M parameters (vs 45M in complex approach)
- Target: <0.5mm point distance (vs 1.5-2.0mm plateau)

Usage:
    from simplified_fps_nps import SimpleFPSNPSConfig, create_simple_fps_nps_model
    
    config = SimpleFPSNPSConfig()
    model = create_simple_fps_nps_model(config.get_model_config())
"""

from config import SimpleFPSNPSConfig, get_config
from network_simple_fps_nps import SimpleFPSNPSNetwork, create_simple_fps_nps_model
from fps_nps_sampling import SimpleFPSSampling, SimpleNPSSampling, CombinedFPSNPSSampling
from simple_losses import SimpleLossFunction, create_simple_loss_function

__version__ = "1.0.0"
__author__ = "TUS-REC Challenge Team"
__description__ = "Simplified FPS/NPS approach for ultrasound pose regression"

__all__ = [
    # Configuration
    'SimpleFPSNPSConfig',
    'get_config',
    
    # Network
    'SimpleFPSNPSNetwork',
    'create_simple_fps_nps_model',
    
    # Sampling
    'SimpleFPSSampling',
    'SimpleNPSSampling', 
    'CombinedFPSNPSSampling',
    
    # Losses
    'SimpleLossFunction',
    'create_simple_loss_function',
]
