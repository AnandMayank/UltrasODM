"""
Configuration for Simplified FPS/NPS Approach

Based on TUS-REC baseline success with optimizations for FPS/NPS.
"""

import os


class SimpleFPSNPSConfig:
    """
    Configuration class for simplified FPS/NPS training
    
    Based on analysis:
    - TUS-REC baseline: 2 frames, batch_size 16, EfficientNet-B1
    - Your complex approach: 7 frames, batch_size 2, 45M parameters
    - Target: Balance between baseline simplicity and FPS/NPS innovation
    """
    
    def __init__(self):
        # Data Configuration (relative to main project directory)
        self.DATA_PATH = '../data/frames_transfs'
        self.FILENAME_CALIB = '../data/calib_matrix.csv'
        self.LANDMARK_PATH = '../data/landmarks'
        
        # Training Configuration (ultra-memory-optimized for GPU-only mode)
        self.NUM_SAMPLES = 2           # Minimal frames for memory (was 3)
        self.SAMPLE_RANGE = 4          # Minimal range for memory (was 6)
        self.MINIBATCH_SIZE = 1        # Single batch for memory (was 2)
        self.NUM_EPOCHS = 1000         # Reasonable epoch count
        self.LEARNING_RATE = 1e-4      # Same as baseline
        
        # Model Configuration
        self.PRED_TYPE = 'parameter'   # Predict 6-DOF parameters
        self.LABEL_TYPE = 'point'      # Use point labels (for PointDistance loss)
        self.NUM_PRED = 1              # Single prediction per sequence
        
        # FPS/NPS Configuration
        self.NUM_FPS_POINTS = 32       # Global motion patterns
        self.NUM_NPS_POINTS = 64       # Local motion details
        self.BACKBONE = 'efficientnet_b1'  # Proven effective backbone

        # Mamba Configuration
        self.MAMBA_STATE_DIM = 16      # State dimension for Mamba layers
        self.USE_MAMBA = True          # Enable/disable Mamba processing
        
        # Loss Configuration (simplified)
        self.LOSS_TYPE = 'point_only'  # Only PointDistance loss
        self.USE_MSE_FALLBACK = True   # MSE when points not available
        
        # Training Configuration
        self.OPTIMIZER = 'adam'
        self.WEIGHT_DECAY = 1e-5
        self.LR_SCHEDULER = 'plateau'
        self.LR_PATIENCE = 10
        self.LR_FACTOR = 0.5
        
        # Validation Configuration
        self.VAL_FREQUENCY = 5         # Validate every 5 epochs
        self.SAVE_FREQUENCY = 50       # Save model every 50 epochs
        self.INFO_FREQUENCY = 10       # Print info every 10 steps
        
        # Data Split Configuration
        self.TRAIN_FOLDS = ['fold_00', 'fold_01', 'fold_02']
        self.VAL_FOLD = 'fold_03'
        self.TEST_FOLD = 'fold_04'
        
        # Output Configuration
        self.SAVE_PATH = 'results/simplified_fps_nps'
        self.MODEL_NAME = 'simple_fps_nps'
        
        # Device Configuration
        self.GPU_IDS = '0'
        self.DEVICE = 'cuda'
        
        # Reproducibility
        self.RANDOM_SEED = 42
        
        # Performance Targets
        self.TARGET_POINT_DISTANCE = 0.5  # mm (better than 1.5-2.0mm current)
        self.TARGET_PARAMETERS = 15e6      # ~15M parameters (vs 45M complex)
        
    def get_model_config(self):
        """Get model-specific configuration"""
        return {
            'input_channels': 1,
            'num_frames': self.NUM_SAMPLES,
            'output_dim': 6,  # 6-DOF parameters
            'num_fps_points': self.NUM_FPS_POINTS,
            'num_nps_points': self.NUM_NPS_POINTS,
            'backbone': self.BACKBONE,
            'mamba_state_dim': self.MAMBA_STATE_DIM,
            'use_mamba': self.USE_MAMBA
        }
    
    def get_training_config(self):
        """Get training-specific configuration"""
        return {
            'batch_size': self.MINIBATCH_SIZE,
            'learning_rate': self.LEARNING_RATE,
            'num_epochs': self.NUM_EPOCHS,
            'optimizer': self.OPTIMIZER,
            'weight_decay': self.WEIGHT_DECAY,
            'lr_scheduler': self.LR_SCHEDULER,
            'lr_patience': self.LR_PATIENCE,
            'lr_factor': self.LR_FACTOR
        }
    
    def get_data_config(self):
        """Get data-specific configuration"""
        return {
            'data_path': self.DATA_PATH,
            'calib_file': self.FILENAME_CALIB,
            'landmark_path': self.LANDMARK_PATH,
            'num_samples': self.NUM_SAMPLES,
            'sample_range': self.SAMPLE_RANGE,
            'pred_type': self.PRED_TYPE,
            'label_type': self.LABEL_TYPE,
            'train_folds': self.TRAIN_FOLDS,
            'val_fold': self.VAL_FOLD,
            'test_fold': self.TEST_FOLD
        }
    
    def get_loss_config(self):
        """Get loss-specific configuration"""
        return {
            'loss_type': self.LOSS_TYPE,
            'use_mse_fallback': self.USE_MSE_FALLBACK
        }
    
    def create_save_directory(self):
        """Create save directory if it doesn't exist"""
        os.makedirs(self.SAVE_PATH, exist_ok=True)
        os.makedirs(os.path.join(self.SAVE_PATH, 'saved_model'), exist_ok=True)
        os.makedirs(os.path.join(self.SAVE_PATH, 'tensorboard'), exist_ok=True)
        return self.SAVE_PATH
    
    def save_config(self, save_path=None):
        """Save configuration to file"""
        if save_path is None:
            save_path = self.SAVE_PATH
        
        config_file = os.path.join(save_path, 'config.txt')
        
        with open(config_file, 'w') as f:
            f.write("# Simplified FPS/NPS Configuration\n")
            f.write("# Based on TUS-REC baseline with FPS/NPS enhancements\n\n")
            
            # Write all configuration attributes
            for attr_name in dir(self):
                if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                    attr_value = getattr(self, attr_name)
                    f.write(f"{attr_name}: {attr_value}\n")
        
        print(f"Configuration saved to: {config_file}")
    
    def print_config(self):
        """Print configuration summary"""
        print("=" * 60)
        print("SIMPLIFIED FPS/NPS CONFIGURATION")
        print("=" * 60)
        print(f"Model: {self.BACKBONE} + FPS/NPS Sampling")
        print(f"Frames: {self.NUM_SAMPLES} (vs baseline: 2, complex: 7)")
        print(f"Batch Size: {self.MINIBATCH_SIZE} (vs baseline: 16, complex: 2)")
        print(f"FPS Points: {self.NUM_FPS_POINTS}, NPS Points: {self.NUM_NPS_POINTS}")
        print(f"Loss: {self.LOSS_TYPE} (PointDistance only)")
        print(f"Target Parameters: ~{self.TARGET_PARAMETERS/1e6:.1f}M (vs complex: 45M)")
        print(f"Target Performance: <{self.TARGET_POINT_DISTANCE}mm point distance")
        print(f"Save Path: {self.SAVE_PATH}")
        print("=" * 60)


# Default configuration instance
default_config = SimpleFPSNPSConfig()


def get_config():
    """Get default configuration"""
    return default_config


def test_config():
    """Test configuration"""
    print("Testing Simplified FPS/NPS Configuration...")
    
    config = SimpleFPSNPSConfig()
    config.print_config()
    
    # Test configuration methods
    model_config = config.get_model_config()
    training_config = config.get_training_config()
    data_config = config.get_data_config()
    loss_config = config.get_loss_config()
    
    print(f"Model config: {model_config}")
    print(f"Training config: {training_config}")
    print(f"Data config keys: {list(data_config.keys())}")
    print(f"Loss config: {loss_config}")
    
    print("[OK] Configuration test passed!")


if __name__ == "__main__":
    test_config()
