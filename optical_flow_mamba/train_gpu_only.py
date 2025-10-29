#!/usr/bin/env python3
"""
GPU-Only Training Script for Enhanced FPS/NPS + UltrasSOM
Forces everything to stay on GPU to avoid CPU/GPU device mismatch issues
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Import from main utils (adjust path for running from simplified_fps_nps)
import sys
sys.path.append('..')
from utils.loader import Dataset
from utils.funs import pair_samples
from utils.plot_functions import read_calib_matrices, reference_image_points
from utils.transform import LabelTransform, PredictionTransform, PointTransform

# Import Enhanced FPS/NPS + UltrasSOM modules
from config import SimpleFPSNPSConfig
from network_fps_nps_real_mamba import create_enhanced_fps_nps_ultrasom_model
from simple_losses import create_enhanced_loss_function


def force_gpu_initialization():
    """Force GPU initialization and ensure everything stays on GPU"""
    
    if not torch.cuda.is_available():
        raise RuntimeError("[ERROR] CUDA not available! This script requires GPU.")
    
    # Force GPU device - no CPU fallback allowed
    device = torch.device('cuda:0')
    
    print("[START] GPU-ONLY MODE: Forcing everything to stay on GPU")
    
    # Aggressive CUDA setup
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Conservative memory settings
    torch.cuda.set_per_process_memory_fraction(0.55)  # Only 55% to be very safe
    
    # Force CUDA context and CUBLAS initialization
    print("[INFO] Initializing CUDA context and CUBLAS...")
    try:
        # Create and test GPU operations
        test_a = torch.randn(8, 8, device=device, dtype=torch.float32)
        test_b = torch.randn(8, 8, device=device, dtype=torch.float32)
        test_c = torch.matmul(test_a, test_b)
        test_inv = torch.linalg.inv(test_c + torch.eye(8, device=device))
        torch.cuda.synchronize()
        
        # Clean up
        del test_a, test_b, test_c, test_inv
        torch.cuda.empty_cache()
        
        print("[OK] CUDA and CUBLAS successfully initialized")
        
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to initialize CUDA/CUBLAS: {e}")
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    print(f"[INFO] GPU Information:")
    print(f"   Device: {device}")
    print(f"   Name: {torch.cuda.get_device_name(0)}")
    print(f"   Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   Memory Fraction: 55%")
    print(f"   Available: {torch.cuda.get_device_properties(0).total_memory * 0.55 / 1e9:.1f} GB")
    
    return device


def setup_data_gpu_only(config, device):
    """Setup data with everything forced to GPU from the start"""
    
    print("[INFO] Setting up data with GPU-only approach...")
    
    # Create dataset
    full_dataset = Dataset(
        data_path=config.DATA_PATH,
        num_samples=config.NUM_SAMPLES,
        sample_range=config.SAMPLE_RANGE
    )
    
    # Split dataset
    dataset_folds = full_dataset.partition_by_ratio(ratios=[1] * 5, randomise=True)
    train_dataset = dataset_folds[0] + dataset_folds[1] + dataset_folds[2]
    val_dataset = dataset_folds[3]
    
    # Create data loaders - minimal workers to reduce memory pressure
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.MINIBATCH_SIZE, shuffle=True,
        num_workers=1, pin_memory=False  # Minimal settings
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.MINIBATCH_SIZE, shuffle=False,
        num_workers=1, pin_memory=False
    )
    
    # Initialize everything on GPU directly
    print("[INFO] Creating tensors directly on GPU...")
    data_pairs = pair_samples(config.NUM_SAMPLES, config.NUM_PRED, 0).to(device)
    
    # Load calibration matrices and move to GPU immediately
    tform_calib_scale, tform_calib_R_T, tform_calib = read_calib_matrices(
        os.path.join(os.getcwd(), config.FILENAME_CALIB)
    )
    tform_calib_scale = tform_calib_scale.to(device)
    tform_calib_R_T = tform_calib_R_T.to(device)
    tform_calib = tform_calib.to(device)
    
    # Get image points and move to GPU
    sample_frame = full_dataset[0][0]
    image_points = reference_image_points(sample_frame.shape[1:], 2).to(device)
    
    print(f"[OK] All tensors created on GPU: {device}")
    print(f"   Data pairs: {data_pairs.shape} on {data_pairs.device}")
    print(f"   Image points: {image_points.shape} on {image_points.device}")
    
    # Create transforms with GPU tensors from the start
    print("[INFO] Creating transforms with GPU tensors...")
    transform_label = LabelTransform(
        config.LABEL_TYPE, pairs=data_pairs, image_points=image_points,
        tform_image_to_tool=tform_calib, tform_image_mm_to_tool=tform_calib_R_T,
        tform_image_pixel_to_mm=tform_calib_scale
    )
    
    transform_prediction = PredictionTransform(
        config.PRED_TYPE, config.LABEL_TYPE, num_pairs=data_pairs.shape[0],
        image_points=image_points, tform_image_to_tool=tform_calib,
        tform_image_mm_to_tool=tform_calib_R_T, tform_image_pixel_to_mm=tform_calib_scale
    )
    
    transform_into_points = PointTransform(
        label_type=config.LABEL_TYPE, image_points=image_points,
        tform_image_to_tool=tform_calib, tform_image_mm_to_tool=tform_calib_R_T,
        tform_image_pixel_to_mm=tform_calib_scale
    )
    
    print("[OK] All transforms created with GPU tensors")
    
    return (train_loader, val_loader, transform_label, 
            transform_prediction, transform_into_points, data_pairs)


def setup_model_gpu_only(config, device, num_pairs):
    """Setup model with GPU-only approach"""
    
    print("[INFO] Creating model directly on GPU...")
    
    # Create model config
    model_config = config.get_model_config()
    model_config.update({
        'num_pairs': num_pairs,
        'mamba_d_state': 64,
        'mamba_d_conv': 4, 
        'mamba_expand': 2,
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 256
    })
    
    # Create model and move to GPU immediately
    model = create_enhanced_fps_nps_ultrasom_model(model_config)
    model = model.to(device)
    
    print(f"[OK] Model created on GPU: {device}")
    print(f"   Model info: {model.get_model_info()}")
    
    # Create loss and optimizer
    criterion = create_enhanced_loss_function('point_focused')
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY, eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=config.LR_PATIENCE,
        factor=config.LR_FACTOR, verbose=True
    )
    
    return model, criterion, optimizer, scheduler


def train_epoch_gpu_only(model, criterion, train_loader, optimizer, device, 
                         transform_label, transform_prediction, transform_into_points, config):
    """Training epoch with GPU-only operations"""
    
    model.train()
    epoch_loss = 0.0
    epoch_distance = 0.0
    num_batches = len(train_loader)
    
    for step, (frames, tforms, _, _) in enumerate(train_loader):
        # Ensure everything is on GPU
        frames = frames.to(device, non_blocking=True)
        tforms = tforms.to(device, non_blocking=True)
        
        # All operations on GPU
        tforms_inv = torch.linalg.inv(tforms)
        frames = frames / 255.0
        
        if step == 0:
            print(f"   [DATA] Training data: frames={frames.device}, tforms={tforms.device}")
        
        # Forward pass - all on GPU
        optimizer.zero_grad()
        labels = transform_label(tforms, tforms_inv)
        model_output = model(frames)
        
        # Handle model output
        outputs = model_output['pose'] if isinstance(model_output, dict) else model_output
        outputs = torch.clamp(outputs, -3.14, 3.14)
        
        # Transform predictions - all on GPU
        preds = transform_prediction(outputs)
        preds_pts = transform_into_points(preds)
        labels_pts = transform_into_points(labels)
        
        # Loss calculation
        loss, loss_dict = criterion(model_output, labels, preds_pts, labels_pts)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        point_distance = ((preds_pts - labels_pts) ** 2).sum(dim=2).sqrt().mean()
        epoch_loss += loss.item()
        epoch_distance += point_distance.item()
        
        # Progress
        if step % config.INFO_FREQUENCY == 0:
            print(f'Step {step:4d}/{num_batches}, Loss: {loss.item():.4f}, '
                  f'Point: {loss_dict.get("point_loss", 0):.4f}, '
                  f'Distance: {point_distance.item():.4f}mm')
    
    return epoch_loss / num_batches, epoch_distance / num_batches


def main():
    """Main GPU-only training function"""
    
    print("[START] GPU-ONLY Enhanced FPS/NPS + UltrasSOM Training")
    print("=" * 60)
    
    # Setup
    config = SimpleFPSNPSConfig()
    config.print_config()
    
    # Force GPU initialization
    device = force_gpu_initialization()
    
    # Create save directory
    config.SAVE_PATH = 'results/gpu_only_ultrasom'
    config.MODEL_NAME = 'gpu_only_ultrasom'
    save_path = config.create_save_directory()
    config.save_config(save_path)
    
    # Setup data with GPU-only approach
    (train_loader, val_loader, transform_label, 
     transform_prediction, transform_into_points, data_pairs) = setup_data_gpu_only(config, device)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Setup model
    model, criterion, optimizer, scheduler = setup_model_gpu_only(config, device, data_pairs.shape[0])
    
    # Training loop
    print("\n[TARGET] Starting GPU-only training...")
    print("=" * 60)
    
    for epoch in range(min(10, config.NUM_EPOCHS)):  # Limit to 10 epochs for testing
        print(f"\nEpoch {epoch + 1}")
        
        train_loss, train_distance = train_epoch_gpu_only(
            model, criterion, train_loader, optimizer, device,
            transform_label, transform_prediction, transform_into_points, config
        )
        
        print(f"Epoch {epoch + 1}: Loss={train_loss:.4f}, Distance={train_distance:.4f}mm")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), 
                      os.path.join(save_path, 'saved_model', f'gpu_only_epoch_{epoch+1}.pth'))
    
    print("[OK] GPU-only training completed successfully!")


if __name__ == "__main__":
    main()
