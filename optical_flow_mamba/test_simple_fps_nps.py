"""
Test Script for Simplified FPS/NPS Implementation

Test all components to ensure they work correctly before training.
"""

import torch
import numpy as np
import sys
import os

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test imports
try:
    from fps_nps_sampling import SimpleFPSSampling, SimpleNPSSampling, CombinedFPSNPSSampling
    from network_simple_fps_nps import SimpleFPSNPSNetwork, create_simple_fps_nps_model
    from simple_losses import SimpleLossFunction, create_simple_loss_function
    from config import SimpleFPSNPSConfig
    print("[OK] All imports successful!")
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    sys.exit(1)


def test_fps_nps_sampling():
    """Test FPS/NPS sampling components"""
    print("\n Testing FPS/NPS Sampling...")
    
    # Test data
    B, T, H, W, C = 2, 4, 32, 32, 256
    features = torch.randn(B, T, H, W, C)
    
    # Test FPS
    fps_sampler = SimpleFPSSampling(num_points=64)
    fps_features, fps_indices = fps_sampler(features)
    assert fps_features.shape == (B, 64, C), f"FPS shape mismatch: {fps_features.shape}"
    print(f"  [OK] FPS sampling: {features.shape} → {fps_features.shape}")
    
    # Test NPS
    nps_sampler = SimpleNPSSampling(num_points=128)
    nps_features, nps_indices = nps_sampler(features)
    assert nps_features.shape == (B, 128, C), f"NPS shape mismatch: {nps_features.shape}"
    print(f"  [OK] NPS sampling: {features.shape} → {nps_features.shape}")
    
    # Test Combined
    combined_sampler = CombinedFPSNPSSampling(num_fps_points=32, num_nps_points=64)
    combined_features, fps_idx, nps_idx = combined_sampler(features)
    assert combined_features.shape == (B, 96, C), f"Combined shape mismatch: {combined_features.shape}"
    print(f"  [OK] Combined sampling: {features.shape} → {combined_features.shape}")
    
    print("[OK] FPS/NPS Sampling tests passed!")


def test_network():
    """Test network architecture"""
    print("\n Testing Network Architecture...")
    
    # Test configuration
    config = {
        'input_channels': 1,
        'num_frames': 4,
        'output_dim': 6,
        'num_fps_points': 32,
        'num_nps_points': 64,
        'backbone': 'efficientnet_b1'
    }
    
    # Create model
    model = create_simple_fps_nps_model(config)
    
    # Test input
    B, T, H, W = 2, 4, 224, 224
    frames = torch.randn(B, T, H, W)
    
    # Forward pass
    output = model(frames)
    assert output.shape == (B, 6), f"Output shape mismatch: {output.shape}"
    print(f"  [OK] Network forward: {frames.shape} → {output.shape}")
    
    # Model info
    info = model.get_model_info()
    print(f"  [OK] Model parameters: {info['total_parameters']:,}")
    print(f"  [OK] Trainable parameters: {info['trainable_parameters']:,}")
    
    # Check parameter count is reasonable (~10-15M target)
    param_count = info['total_parameters']
    if param_count > 50e6:
        print(f"  [WARNING]  Warning: Model has {param_count/1e6:.1f}M parameters (target: ~10-15M)")
    else:
        print(f"  [OK] Parameter count reasonable: {param_count/1e6:.1f}M")
    
    print("[OK] Network Architecture tests passed!")


def test_losses():
    """Test loss functions"""
    print("\n Testing Loss Functions...")
    
    # Test data
    B, N = 4, 5
    predictions = torch.randn(B, 6)
    targets = torch.randn(B, 6)
    pred_points = torch.randn(B, N, 3)
    target_points = torch.randn(B, N, 3)
    
    # Test SimpleLossFunction
    simple_loss = SimpleLossFunction()
    loss1, loss_dict1 = simple_loss(predictions, targets, pred_points, target_points)
    assert isinstance(loss1, torch.Tensor), "Loss should be tensor"
    assert 'total_loss' in loss_dict1, "Loss dict should contain total_loss"
    print(f"  [OK] SimpleLossFunction: {loss1.item():.4f}")
    
    # Test factory function
    point_only_loss = create_simple_loss_function('point_only')
    loss2, loss_dict2 = point_only_loss(predictions, targets, pred_points, target_points)
    print(f"  [OK] Factory point_only: {loss2.item():.4f}")
    
    # Test without points (MSE fallback)
    loss3, loss_dict3 = simple_loss(predictions, targets, None, None)
    assert loss_dict3['loss_type'] == 'mse_fallback', "Should use MSE fallback"
    print(f"  [OK] MSE fallback: {loss3.item():.4f}")
    
    print("[OK] Loss Functions tests passed!")


def test_config():
    """Test configuration"""
    print("\n Testing Configuration...")
    
    config = SimpleFPSNPSConfig()
    
    # Test configuration methods
    model_config = config.get_model_config()
    training_config = config.get_training_config()
    data_config = config.get_data_config()
    loss_config = config.get_loss_config()
    
    # Check required keys
    assert 'num_frames' in model_config, "Model config missing num_frames"
    assert 'batch_size' in training_config, "Training config missing batch_size"
    assert 'data_path' in data_config, "Data config missing data_path"
    assert 'loss_type' in loss_config, "Loss config missing loss_type"
    
    print(f"  [OK] Model config: {len(model_config)} keys")
    print(f"  [OK] Training config: {len(training_config)} keys")
    print(f"  [OK] Data config: {len(data_config)} keys")
    print(f"  [OK] Loss config: {len(loss_config)} keys")
    
    # Test target parameters
    target_params = config.TARGET_PARAMETERS
    target_distance = config.TARGET_POINT_DISTANCE
    print(f"  [OK] Target parameters: {target_params/1e6:.1f}M")
    print(f"  [OK] Target distance: {target_distance}mm")
    
    print("[OK] Configuration tests passed!")


def test_integration():
    """Test integration of all components"""
    print("\n Testing Integration...")
    
    # Create config
    config = SimpleFPSNPSConfig()
    model_config = config.get_model_config()
    
    # Create model
    model = create_simple_fps_nps_model(model_config)
    
    # Create loss function
    criterion = create_simple_loss_function('point_only')
    
    # Test data
    B, T, H, W = 2, 4, 224, 224
    frames = torch.randn(B, T, H, W)

    # Forward pass
    outputs = model(frames)

    # Mock transformed points for loss calculation
    # Use model outputs to ensure gradient connection
    N = 5
    pred_points = outputs.unsqueeze(1).expand(-1, N, -1)[:, :, :3]  # Use model output
    target_points = torch.randn(B, N, 3)
    targets = torch.randn(B, 6)

    # Calculate loss (should be connected to model now)
    loss, loss_dict = criterion(outputs, targets, pred_points, target_points)

    # Test backward pass
    loss.backward()
    
    print(f"  [OK] Forward pass: {frames.shape} → {outputs.shape}")
    print(f"  [OK] Loss calculation: {loss.item():.4f}")
    print(f"  [OK] Backward pass successful")
    
    # Check gradients
    has_gradients = any(p.grad is not None for p in model.parameters())
    assert has_gradients, "Model should have gradients after backward pass"
    print(f"  [OK] Gradients computed")
    
    print("[OK] Integration tests passed!")


def main():
    """Run all tests"""
    print("[START] Testing Simplified FPS/NPS Implementation")
    print("=" * 60)
    
    try:
        test_fps_nps_sampling()
        test_network()
        test_losses()
        test_config()
        test_integration()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("[OK] Simplified FPS/NPS implementation is ready for training")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
