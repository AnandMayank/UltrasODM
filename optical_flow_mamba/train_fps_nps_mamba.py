"""
FPS/NPS + Mamba Training Script

Enhanced training script for FPS/NPS + Mamba architecture.
Combines the proven EfficientNet backbone with FPS/NPS sampling and Mamba processing.
"""

import os
import sys
import torch
import json
import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from main utils (borrowed)
from utils.loader import Dataset
from utils.funs import pair_samples
from utils.plot_functions import read_calib_matrices, reference_image_points
from utils.transform import LabelTransform, PredictionTransform, PointTransform

# Import FPS/NPS + Mamba modules
from config import SimpleFPSNPSConfig
from network_fps_nps_mamba import create_fps_nps_mamba_model
from simple_losses import create_simple_loss_function


def setup_training(config):
    """Setup training environment"""
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Setup device
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_IDS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create save directory
    save_path = config.create_save_directory()
    
    # Update save path for Mamba version
    config.SAVE_PATH = 'results/simplified_fps_nps_mamba'
    config.MODEL_NAME = 'fps_nps_mamba'
    save_path = config.create_save_directory()
    
    # Save configuration
    config.save_config(save_path)
    
    return device, save_path


def setup_data(config, device):
    """Setup data loaders and transformations"""
    
    # Create dataset
    full_dataset = Dataset(
        data_path=config.DATA_PATH,
        num_samples=config.NUM_SAMPLES,
        sample_range=config.SAMPLE_RANGE
    )
    
    # Split dataset
    dataset_folds = full_dataset.partition_by_ratio(
        ratios=[1] * 5,
        randomise=True
    )
    
    # Create train/val datasets
    train_dataset = dataset_folds[0] + dataset_folds[1] + dataset_folds[2]
    val_dataset = dataset_folds[3]
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.MINIBATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.MINIBATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup transformations
    data_pairs = pair_samples(config.NUM_SAMPLES, config.NUM_PRED, 0).to(device)
    
    # Load calibration matrices
    tform_calib_scale, tform_calib_R_T, tform_calib = read_calib_matrices(
        os.path.join(os.getcwd(), config.FILENAME_CALIB)
    )
    
    # Get image points
    sample_frame = full_dataset[0][0]
    image_points = reference_image_points(sample_frame.shape[1:], 2).to(device)
    
    # Create transformation functions
    transform_label = LabelTransform(
        config.LABEL_TYPE,
        pairs=data_pairs,
        image_points=image_points,
        tform_image_to_tool=tform_calib.to(device),
        tform_image_mm_to_tool=tform_calib_R_T.to(device),
        tform_image_pixel_to_mm=tform_calib_scale.to(device)
    )
    
    transform_prediction = PredictionTransform(
        config.PRED_TYPE,
        config.LABEL_TYPE,
        num_pairs=data_pairs.shape[0],
        image_points=image_points,
        tform_image_to_tool=tform_calib.to(device),
        tform_image_mm_to_tool=tform_calib_R_T.to(device),
        tform_image_pixel_to_mm=tform_calib_scale.to(device)
    )
    
    transform_into_points = PointTransform(
        label_type=config.LABEL_TYPE,
        image_points=image_points,
        tform_image_to_tool=tform_calib.to(device),
        tform_image_mm_to_tool=tform_calib_R_T.to(device),
        tform_image_pixel_to_mm=tform_calib_scale.to(device)
    )
    
    return (train_loader, val_loader, transform_label, 
            transform_prediction, transform_into_points, data_pairs)


def setup_model_and_optimizer(config, device, num_pairs):
    """Setup model, loss function, and optimizer"""
    
    # Create model
    model_config = config.get_model_config()
    model_config['num_pairs'] = num_pairs  # Add num_pairs to config
    model_config['mamba_state_dim'] = 16   # Add Mamba configuration
    model = create_fps_nps_mamba_model(model_config).to(device)
    
    # Print model info
    model_info = model.get_model_info()
    print(f"FPS/NPS + Mamba Model created: {model_info}")
    
    # Create loss function
    loss_config = config.get_loss_config()
    criterion = create_simple_loss_function(loss_config['loss_type'])
    
    # Create optimizer
    training_config = config.get_training_config()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=training_config['lr_patience'],
        factor=training_config['lr_factor'],
        verbose=True
    )
    
    return model, criterion, optimizer, scheduler


def train_epoch(model, criterion, train_loader, optimizer, device,
                transform_label, transform_prediction, transform_into_points, config):
    """Train for one epoch"""
    
    model.train()
    epoch_loss = 0.0
    epoch_distance = 0.0
    num_batches = len(train_loader)
    
    for step, (frames, tforms, _, _) in enumerate(train_loader):
        frames, tforms = frames.to(device), tforms.to(device)
        tforms_inv = torch.linalg.inv(tforms)
        frames = frames / 255.0  # Normalize
        
        # Transform labels
        labels = transform_label(tforms, tforms_inv)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(frames)
        
        # Debug: Check for invalid outputs
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print(f"Warning: Invalid outputs detected at step {step}")
            continue
        
        # Clamp outputs to reasonable range
        outputs = torch.clamp(outputs, -3.14, 3.14)
        
        # Transform predictions
        preds = transform_prediction(outputs)
        
        # Transform to points for loss calculation
        preds_pts = transform_into_points(preds)
        labels_pts = transform_into_points(labels)
        
        # Calculate loss
        loss, loss_dict = criterion(preds, labels, preds_pts, labels_pts)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate point distance for monitoring
        point_distance = ((preds_pts - labels_pts) ** 2).sum(dim=2).sqrt().mean()
        
        # Accumulate metrics
        epoch_loss += loss.item()
        epoch_distance += point_distance.item()
        
        # Print progress
        if step % config.INFO_FREQUENCY == 0:
            print(f'Step {step:4d}/{num_batches}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Distance: {point_distance.item():.4f}mm')
    
    return epoch_loss / num_batches, epoch_distance / num_batches


def validate_epoch(model, criterion, val_loader, device,
                   transform_label, transform_prediction, transform_into_points):
    """Validate for one epoch"""
    
    model.eval()
    epoch_loss = 0.0
    epoch_distance = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for frames, tforms, _, _ in val_loader:
            frames, tforms = frames.to(device), tforms.to(device)
            tforms_inv = torch.linalg.inv(tforms)
            frames = frames / 255.0
            
            # Transform labels
            labels = transform_label(tforms, tforms_inv)
            
            # Forward pass
            outputs = model(frames)
            
            # Clamp outputs
            outputs = torch.clamp(outputs, -3.14, 3.14)
            
            preds = transform_prediction(outputs)
            
            # Transform to points
            preds_pts = transform_into_points(preds)
            labels_pts = transform_into_points(labels)
            
            # Calculate loss
            loss, loss_dict = criterion(preds, labels, preds_pts, labels_pts)
            
            # Calculate point distance
            point_distance = ((preds_pts - labels_pts) ** 2).sum(dim=2).sqrt().mean()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_distance += point_distance.item()
    
    return epoch_loss / num_batches, epoch_distance / num_batches


def main():
    """Main training function"""
    
    print("[START] Starting FPS/NPS + Mamba Training")
    print("=" * 60)
    
    # Setup configuration
    config = SimpleFPSNPSConfig()
    print("[INFO] Enhanced with Mamba layers for better temporal modeling")
    config.print_config()
    
    # Setup training environment
    device, save_path = setup_training(config)
    print(f"Device: {device}")
    print(f"Save path: {save_path}")
    
    # Setup data
    (train_loader, val_loader, transform_label, 
     transform_prediction, transform_into_points, data_pairs) = setup_data(config, device)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Setup model and optimizer
    num_pairs = data_pairs.shape[0]
    model, criterion, optimizer, scheduler = setup_model_and_optimizer(config, device, num_pairs)
    
    # Setup TensorBoard
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    
    # Training loop
    best_val_distance = float('inf')
    
    print("\n[TARGET] Starting FPS/NPS + Mamba training loop...")
    print(f"Target: <{config.TARGET_POINT_DISTANCE}mm point distance")
    print("=" * 60)
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        
        # Train
        train_loss, train_distance = train_epoch(
            model, criterion, train_loader, optimizer, device,
            transform_label, transform_prediction, transform_into_points, config
        )
        
        print(f"Train - Loss: {train_loss:.4f}, Distance: {train_distance:.4f}mm")
        
        # Validate
        if (epoch + 1) % config.VAL_FREQUENCY == 0:
            val_loss, val_distance = validate_epoch(
                model, criterion, val_loader, device,
                transform_label, transform_prediction, transform_into_points
            )
            
            print(f"Val   - Loss: {val_loss:.4f}, Distance: {val_distance:.4f}mm")
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Log to TensorBoard
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Distance/Train', train_distance, epoch)
            writer.add_scalar('Distance/Val', val_distance, epoch)
            
            # Save best model
            if val_distance < best_val_distance:
                best_val_distance = val_distance
                torch.save(model.state_dict(), 
                          os.path.join(save_path, 'saved_model', 'best_model.pth'))
                print(f"[OK] New best FPS/NPS + Mamba model saved! Distance: {val_distance:.4f}mm")
        
        # Save checkpoint
        if (epoch + 1) % config.SAVE_FREQUENCY == 0:
            torch.save(model.state_dict(), 
                      os.path.join(save_path, 'saved_model', f'model_epoch_{epoch+1:04d}.pth'))
    
    print(f"\n[SUCCESS] FPS/NPS + Mamba training completed!")
    print(f"Best validation distance: {best_val_distance:.4f}mm")
    print(f"Target was: <{config.TARGET_POINT_DISTANCE}mm")
    
    writer.close()


if __name__ == "__main__":
    main()
