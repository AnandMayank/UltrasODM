"""
Enhanced FPS/NPS + UltrasSOM Training Script

Training script for Enhanced FPS/NPS + UltrasSOM architecture with:
- Video patch embedding with adjustable windows
- Optical flow integration for motion dynamics
- Enhanced multi-component loss functions
- Bidirectional Mamba with selective scan
- Target: <0.2mm clinical accuracy
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

# Import Enhanced FPS/NPS + UltrasSOM modules
from config import SimpleFPSNPSConfig
from network_fps_nps_real_mamba import create_enhanced_fps_nps_ultrasom_model
from simple_losses import create_enhanced_loss_function


def setup_training(config):
    """Setup training environment"""
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Setup device with proper CUDA initialization
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_IDS

    if torch.cuda.is_available():
        # Complete CUDA reset and initialization
        torch.cuda.empty_cache()

        # Initialize CUDA context
        device = torch.device('cuda')

        # Force CUDA context creation with a simple operation
        try:
            dummy_tensor = torch.zeros(1).cuda()
            del dummy_tensor
            torch.cuda.synchronize()
        except Exception as e:
            print(f"[WARNING]  CUDA context creation failed: {e}")
            device = torch.device('cpu')
            print("Falling back to CPU")

        if device.type == 'cuda':
            # Set memory fraction to avoid OOM
            torch.cuda.set_per_process_memory_fraction(0.70)  # Further reduced to 70%

            # Initialize CUBLAS explicitly
            try:
                # Force CUBLAS initialization with matrix operations
                test_matrix = torch.randn(4, 4, device=device)
                test_inv = torch.linalg.inv(test_matrix)
                torch.cuda.synchronize()
                del test_matrix, test_inv
                print("[OK] CUBLAS successfully initialized")
            except Exception as e:
                print(f"[WARNING]  CUBLAS initialization failed: {e}")
                print("[INFO] Attempting CUBLAS recovery...")
                torch.cuda.empty_cache()
                # Try again with smaller test
                try:
                    test_matrix = torch.eye(2, device=device)
                    test_inv = torch.linalg.inv(test_matrix)
                    torch.cuda.synchronize()
                    del test_matrix, test_inv
                    print("[OK] CUBLAS recovery successful")
                except Exception as e2:
                    print(f"[ERROR] CUBLAS recovery failed: {e2}")
                    device = torch.device('cpu')
                    print("[INFO] Falling back to CPU")

            if device.type == 'cuda':
                # Enable memory optimization
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False  # For performance

        print(f"[INFO] Device setup:")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        print(f"   CUDA device count: {torch.cuda.device_count()}")
        print(f"   Current CUDA device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   Memory fraction set to 80%")
        print(f"   Using device: {device}")
    else:
        device = torch.device('cpu')
        print("[WARNING]  CUDA not available, using CPU")
    
    # Update save path for Enhanced UltrasSOM version
    config.SAVE_PATH = 'results/enhanced_fps_nps_ultrasom'
    config.MODEL_NAME = 'enhanced_fps_nps_ultrasom'
    save_path = config.create_save_directory()
    
    # Save configuration
    config.save_config(save_path)
    
    return device, save_path


def setup_data(config, device):
    """Setup data loaders and transformations with robust CUDA handling"""

    print(f"[INFO] Setting up data with device: {device}")

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
    
    # Create data loaders with memory optimization
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.MINIBATCH_SIZE,
        shuffle=True,
        num_workers=2,  # Reduced from 4 to save memory
        pin_memory=False  # Disabled to reduce CUDA memory pressure
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.MINIBATCH_SIZE,
        shuffle=False,
        num_workers=2,  # Reduced from 4 to save memory
        pin_memory=False  # Disabled to reduce CUDA memory pressure
    )
    
    # Setup transformations - Initialize on CPU first to avoid CUDA issues
    print("[INFO] Initializing data pairs and calibration matrices...")
    data_pairs = pair_samples(config.NUM_SAMPLES, config.NUM_PRED, 0)

    # Load calibration matrices (keep on CPU initially)
    tform_calib_scale, tform_calib_R_T, tform_calib = read_calib_matrices(
        os.path.join(os.getcwd(), config.FILENAME_CALIB)
    )

    # Get image points (keep on CPU initially)
    sample_frame = full_dataset[0][0]
    image_points = reference_image_points(sample_frame.shape[1:], 2)

    print(f"[INFO] Data pairs shape: {data_pairs.shape}")
    print(f"[INFO] Image points shape: {image_points.shape}")
    
    # Create transformation functions - Initialize on CPU to avoid CUDA issues
    print("[INFO] Initializing transformation functions on CPU...")
    transform_label = LabelTransform(
        config.LABEL_TYPE,
        pairs=data_pairs,
        image_points=image_points,
        tform_image_to_tool=tform_calib,
        tform_image_mm_to_tool=tform_calib_R_T,
        tform_image_pixel_to_mm=tform_calib_scale
    )
    
    transform_prediction = PredictionTransform(
        config.PRED_TYPE,
        config.LABEL_TYPE,
        num_pairs=data_pairs.shape[0],
        image_points=image_points,
        tform_image_to_tool=tform_calib,
        tform_image_mm_to_tool=tform_calib_R_T,
        tform_image_pixel_to_mm=tform_calib_scale
    )

    transform_into_points = PointTransform(
        label_type=config.LABEL_TYPE,
        image_points=image_points,
        tform_image_to_tool=tform_calib,
        tform_image_mm_to_tool=tform_calib_R_T,
        tform_image_pixel_to_mm=tform_calib_scale
    )

    # Now move tensors to GPU if available
    if device.type == 'cuda':
        print("[INFO] Moving tensors and transforms to GPU...")
        data_pairs = data_pairs.to(device)
        image_points = image_points.to(device)

        # Move transform internal tensors to GPU
        def move_transform_to_device(transform_obj, device):
            """Move all tensors in a transform object to the specified device"""
            for attr_name in dir(transform_obj):
                if not attr_name.startswith('_'):
                    attr_value = getattr(transform_obj, attr_name)
                    if isinstance(attr_value, torch.Tensor):
                        setattr(transform_obj, attr_name, attr_value.to(device))

        move_transform_to_device(transform_label, device)
        move_transform_to_device(transform_prediction, device)
        move_transform_to_device(transform_into_points, device)

        print("[OK] Tensors and transforms successfully moved to GPU")
    
    return (train_loader, val_loader, transform_label, 
            transform_prediction, transform_into_points, data_pairs)


def setup_model_and_optimizer(config, device, num_pairs):
    """Setup enhanced model, loss function, and optimizer"""

    # Create enhanced model
    model_config = config.get_model_config()
    model_config['num_pairs'] = num_pairs

    # Add Enhanced UltrasSOM specific configuration
    model_config['mamba_d_state'] = 64
    model_config['mamba_d_conv'] = 4
    model_config['mamba_expand'] = 2
    model_config['img_size'] = 224
    model_config['patch_size'] = 16
    model_config['embed_dim'] = 256

    model = create_enhanced_fps_nps_ultrasom_model(model_config)

    # Move model to device (GPU if available)
    if device.type == 'cuda':
        model = model.cuda()
        print(f"[OK] Model moved to CUDA device: {device}")
    else:
        model = model.to(device)
        print(f"[WARNING]  Model running on CPU: {device}")

    # Print model info
    model_info = model.get_model_info()
    print(f"Enhanced UltrasSOM Model created: {model_info}")

    # Create enhanced loss function - Use point-focused for clinical accuracy
    loss_config = config.get_loss_config()
    # Use point-focused loss to prioritize clinical accuracy over other metrics
    criterion = create_enhanced_loss_function('point_focused')
    print("[TARGET] Using point-focused loss configuration:")
    print("   - Point Loss Weight: 10.0 (dominant)")
    print("   - MSE Loss Weight: 0.05 (minimal)")
    print("   - Correlation Loss Weight: 0.02 (minimal)")
    print("   - Velocity Loss Weight: 0.02 (minimal)")

    # Create optimizer with enhanced settings
    training_config = config.get_training_config()
    optimizer = torch.optim.AdamW(  # AdamW for better generalization
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Create learning rate scheduler with warmup
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

    # Initialize loss component accumulators
    epoch_loss_components = {
        'total_loss': 0.0,
        'mse_loss': 0.0,
        'correlation_loss': 0.0,
        'velocity_loss': 0.0,
        'point_loss': 0.0
    }
    
    for step, (frames, tforms, _, _) in enumerate(train_loader):
        # Move data to device (GPU if available)
        frames = frames.to(device, non_blocking=True)
        tforms = tforms.to(device, non_blocking=True)
        tforms_inv = torch.linalg.inv(tforms)
        frames = frames / 255.0  # Normalize

        # Debug: Print device info for first batch
        if step == 0:
            print(f"   [DATA] Data on device: frames={frames.device}, tforms={tforms.device}")
        
        # Transform labels
        labels = transform_label(tforms, tforms_inv)
        
        # Forward pass
        optimizer.zero_grad()
        model_output = model(frames)

        # Handle enhanced model output format
        if isinstance(model_output, dict):
            outputs = model_output['pose']
            motion_info = model_output.get('motion_info', {})
        else:
            outputs = model_output
            motion_info = {}

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

        # Calculate enhanced loss with motion information
        loss, loss_dict = criterion(model_output, labels, preds_pts, labels_pts)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate point distance for monitoring
        point_distance = ((preds_pts - labels_pts) ** 2).sum(dim=2).sqrt().mean()

        # Accumulate metrics
        epoch_loss += loss.item()
        epoch_distance += point_distance.item()

        # Accumulate loss components
        for key in epoch_loss_components:
            if key in loss_dict:
                epoch_loss_components[key] += loss_dict[key]

        # Print progress with loss components
        if step % config.INFO_FREQUENCY == 0:
            print(f'Step {step:4d}/{num_batches}, '
                  f'Total: {loss.item():.4f}, '
                  f'Point: {loss_dict.get("point_loss", 0):.4f}, '
                  f'MSE: {loss_dict.get("mse_loss", 0):.4f}, '
                  f'Corr: {loss_dict.get("correlation_loss", 0):.4f}, '
                  f'Vel: {loss_dict.get("velocity_loss", 0):.4f}, '
                  f'Dist: {point_distance.item():.4f}mm')

    # Average loss components
    for key in epoch_loss_components:
        epoch_loss_components[key] /= num_batches

    return epoch_loss / num_batches, epoch_distance / num_batches, epoch_loss_components


def validate_epoch(model, criterion, val_loader, device,
                   transform_label, transform_prediction, transform_into_points):
    """Validate for one epoch"""

    model.eval()
    epoch_loss = 0.0
    epoch_distance = 0.0
    num_batches = len(val_loader)

    # Initialize loss component accumulators
    epoch_loss_components = {
        'total_loss': 0.0,
        'mse_loss': 0.0,
        'correlation_loss': 0.0,
        'velocity_loss': 0.0,
        'point_loss': 0.0
    }
    
    with torch.no_grad():
        for frames, tforms, _, _ in val_loader:
            # Move data to device (GPU if available)
            frames = frames.to(device, non_blocking=True)
            tforms = tforms.to(device, non_blocking=True)
            tforms_inv = torch.linalg.inv(tforms)
            frames = frames / 255.0
            
            # Transform labels
            labels = transform_label(tforms, tforms_inv)
            
            # Forward pass
            model_output = model(frames)

            # Handle enhanced model output format
            if isinstance(model_output, dict):
                outputs = model_output['pose']
            else:
                outputs = model_output

            # Clamp outputs
            outputs = torch.clamp(outputs, -3.14, 3.14)

            preds = transform_prediction(outputs)

            # Transform to points
            preds_pts = transform_into_points(preds)
            labels_pts = transform_into_points(labels)

            # Calculate enhanced loss
            loss, loss_dict = criterion(model_output, labels, preds_pts, labels_pts)
            
            # Calculate point distance
            point_distance = ((preds_pts - labels_pts) ** 2).sum(dim=2).sqrt().mean()

            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_distance += point_distance.item()

            # Accumulate loss components
            for key in epoch_loss_components:
                if key in loss_dict:
                    epoch_loss_components[key] += loss_dict[key]

    # Average loss components
    for key in epoch_loss_components:
        epoch_loss_components[key] /= num_batches

    return epoch_loss / num_batches, epoch_distance / num_batches, epoch_loss_components


def main():
    """Main training function"""
    
    print("[START] Starting Enhanced FPS/NPS + UltrasSOM Training")
    print("=" * 70)

    # Setup configuration
    config = SimpleFPSNPSConfig()
    print("[INFO] Enhanced with UltrasSOM Architecture:")
    print("   - Video Patch Embedding with Adjustable Windows")
    print("   - Optical Flow Integration for Motion Dynamics")
    print("   - Bidirectional Mamba with Selective Scan")
    print("   - Multi-Component Loss Functions")
    print("   - Target: <0.2mm Clinical Accuracy")
    print("[INFO] Memory-Optimized Configuration:")
    print(f"   - Frames: {config.NUM_SAMPLES} (reduced for memory)")
    print(f"   - Batch Size: {config.MINIBATCH_SIZE} (reduced for memory)")
    print(f"   - Sample Range: {config.SAMPLE_RANGE} (reduced for memory)")
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
    
    print("\n[TARGET] Starting Enhanced UltrasSOM training loop...")
    print(f"Target: <0.2mm clinical accuracy (config target: {config.TARGET_POINT_DISTANCE}mm)")
    print("=" * 70)
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        
        # Train
        train_loss, train_distance, train_loss_components = train_epoch(
            model, criterion, train_loader, optimizer, device,
            transform_label, transform_prediction, transform_into_points, config
        )

        print(f"Train - Total: {train_loss:.4f}, Point: {train_loss_components['point_loss']:.4f}, "
              f"MSE: {train_loss_components['mse_loss']:.4f}, "
              f"Corr: {train_loss_components['correlation_loss']:.4f}, "
              f"Vel: {train_loss_components['velocity_loss']:.4f}, "
              f"Distance: {train_distance:.4f}mm")

        # Validate
        if (epoch + 1) % config.VAL_FREQUENCY == 0:
            val_loss, val_distance, val_loss_components = validate_epoch(
                model, criterion, val_loader, device,
                transform_label, transform_prediction, transform_into_points
            )

            print(f"Val   - Total: {val_loss:.4f}, Point: {val_loss_components['point_loss']:.4f}, "
                  f"MSE: {val_loss_components['mse_loss']:.4f}, "
                  f"Corr: {val_loss_components['correlation_loss']:.4f}, "
                  f"Vel: {val_loss_components['velocity_loss']:.4f}, "
                  f"Distance: {val_distance:.4f}mm")

            # Update learning rate
            scheduler.step(val_loss)

            # Log to TensorBoard - Main metrics
            writer.add_scalar('Loss/Train_Total', train_loss, epoch)
            writer.add_scalar('Loss/Val_Total', val_loss, epoch)
            writer.add_scalar('Distance/Train', train_distance, epoch)
            writer.add_scalar('Distance/Val', val_distance, epoch)

            # Log individual loss components - Training
            writer.add_scalar('Loss_Components/Train_MSE', train_loss_components['mse_loss'], epoch)
            writer.add_scalar('Loss_Components/Train_Correlation', train_loss_components['correlation_loss'], epoch)
            writer.add_scalar('Loss_Components/Train_Velocity', train_loss_components['velocity_loss'], epoch)
            writer.add_scalar('Loss_Components/Train_Point', train_loss_components['point_loss'], epoch)

            # Log individual loss components - Validation
            writer.add_scalar('Loss_Components/Val_MSE', val_loss_components['mse_loss'], epoch)
            writer.add_scalar('Loss_Components/Val_Correlation', val_loss_components['correlation_loss'], epoch)
            writer.add_scalar('Loss_Components/Val_Velocity', val_loss_components['velocity_loss'], epoch)
            writer.add_scalar('Loss_Components/Val_Point', val_loss_components['point_loss'], epoch)

            # Log loss ratios for analysis
            if train_loss > 0:
                writer.add_scalar('Loss_Ratios/MSE_to_Total', train_loss_components['mse_loss'] / train_loss, epoch)
                writer.add_scalar('Loss_Ratios/Correlation_to_Total', train_loss_components['correlation_loss'] / train_loss, epoch)
                writer.add_scalar('Loss_Ratios/Velocity_to_Total', train_loss_components['velocity_loss'] / train_loss, epoch)
                writer.add_scalar('Loss_Ratios/Point_to_Total', train_loss_components['point_loss'] / train_loss, epoch)

            # Log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Training/Learning_Rate', current_lr, epoch)

            # Log performance metrics
            writer.add_scalar('Performance/Train_Val_Loss_Ratio', train_loss / val_loss if val_loss > 0 else 0, epoch)
            writer.add_scalar('Performance/Train_Val_Distance_Ratio', train_distance / val_distance if val_distance > 0 else 0, epoch)

            # Log clinical accuracy progress
            clinical_target = 0.2  # 0.2mm target
            writer.add_scalar('Clinical/Distance_to_Target_mm', val_distance - clinical_target, epoch)
            writer.add_scalar('Clinical/Target_Achievement_Ratio', clinical_target / val_distance if val_distance > 0 else 0, epoch)
            
            # Save best model
            if val_distance < best_val_distance:
                best_val_distance = val_distance
                torch.save(model.state_dict(), 
                          os.path.join(save_path, 'saved_model', 'best_model.pth'))
                print(f"[OK] New best Enhanced UltrasSOM model saved! Distance: {val_distance:.4f}mm")
        
        # Save checkpoint
        if (epoch + 1) % config.SAVE_FREQUENCY == 0:
            torch.save(model.state_dict(), 
                      os.path.join(save_path, 'saved_model', f'model_epoch_{epoch+1:04d}.pth'))
    
    print(f"\n[SUCCESS] Enhanced UltrasSOM training completed!")
    print(f"Best validation distance: {best_val_distance:.4f}mm")
    print(f"Clinical target: <0.2mm (config target: {config.TARGET_POINT_DISTANCE}mm)")

    if best_val_distance < 0.2:
        print("[ACHIEVEMENT] CLINICAL ACCURACY ACHIEVED! <0.2mm target reached!")
    elif best_val_distance < 0.5:
        print("[TARGET] Excellent accuracy achieved! Close to clinical target.")
    elif best_val_distance < 1.0:
        print("[OK] Good accuracy achieved! Further optimization recommended.")
    else:
        print("[WARNING]  Accuracy needs improvement. Consider hyperparameter tuning.")
    
    writer.close()


if __name__ == "__main__":
    main()
