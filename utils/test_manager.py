"""
Test Directory Management Utility
Automatically creates and manages test directories for different training scripts
"""

import os
import glob
import json
import datetime
from typing import Dict, Any, Optional


class TestManager:
    """
    Manages automatic test directory creation and metrics tracking
    """
    
    def __init__(self, script_name: str, base_results_dir: str = "results"):
        """
        Initialize TestManager for a specific script
        
        Args:
            script_name: Name of the training script (e.g., 'train', 'train_1')
            base_results_dir: Base directory for all results
        """
        self.script_name = script_name
        self.base_results_dir = base_results_dir
        self.script_prefix = f"{script_name}_test"
        
    def get_next_test_number(self) -> int:
        """
        Find the next available test number for this script
        
        Returns:
            Next available test number
        """
        pattern = os.path.join(os.getcwd(), self.base_results_dir, f"{self.script_prefix}_*")
        test_dirs = glob.glob(pattern)
        
        if not test_dirs:
            return 1
        
        # Extract test numbers from directory names
        test_numbers = []
        for test_dir in test_dirs:
            dir_name = os.path.basename(test_dir)
            if dir_name.startswith(f"{self.script_prefix}_"):
                try:
                    test_num = int(dir_name.split('_')[-1])
                    test_numbers.append(test_num)
                except (ValueError, IndexError):
                    continue
        
        return max(test_numbers) + 1 if test_numbers else 1
    
    def create_test_directory(self, test_number: int, opt: Any, 
                            loss_config: Optional[Dict] = None) -> str:
        """
        Create a new test directory with the given test number
        
        Args:
            test_number: Test number to create
            opt: Training options object
            loss_config: Loss configuration dictionary
            
        Returns:
            Path to created test directory
        """
        test_dir_name = f"{self.script_prefix}_{test_number}"
        test_path = os.path.join(os.getcwd(), self.base_results_dir, test_dir_name)
        
        # Create main test directory
        os.makedirs(test_path, exist_ok=True)
        
        # Create subdirectories
        subdirs = ['saved_model', 'train_results', 'val_results', 'metrics', 'logs']
        for subdir in subdirs:
            os.makedirs(os.path.join(test_path, subdir), exist_ok=True)
        
        # Save test configuration
        config_data = {
            'script_name': self.script_name,
            'test_number': test_number,
            'timestamp': datetime.datetime.now().isoformat(),
            'model_config': {
                'model_name': getattr(opt, 'model_name', 'unknown'),
                'num_samples': getattr(opt, 'NUM_SAMPLES', None),
                'sample_range': getattr(opt, 'SAMPLE_RANGE', None),
                'num_pred': getattr(opt, 'NUM_PRED', None),
                'pred_type': getattr(opt, 'PRED_TYPE', None),
                'label_type': getattr(opt, 'LABEL_TYPE', None),
            },
            'training_config': {
                'batch_size': getattr(opt, 'MINIBATCH_SIZE', None),
                'learning_rate': getattr(opt, 'LEARNING_RATE', None),
                'num_epochs': getattr(opt, 'NUM_EPOCHS', None),
                'retrain': getattr(opt, 'retrain', False),
                'retrain_epoch': getattr(opt, 'retrain_epoch', '00000000'),
            },
            'data_config': {
                'data_path': getattr(opt, 'DATA_PATH', None),
                'calib_file': getattr(opt, 'FILENAME_CALIB', None),
                'landmark_path': getattr(opt, 'LANDMARK_PATH', None),
            }
        }
        
        # Add loss configuration if provided
        if loss_config:
            config_data['loss_config'] = loss_config
        
        # Save configuration as JSON
        config_file = os.path.join(test_path, 'test_config.json')
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        # Save configuration as readable text
        config_txt_file = os.path.join(test_path, 'test_config.txt')
        with open(config_txt_file, 'w') as f:
            f.write(f"=== {self.script_name.upper()} Test {test_number} Configuration ===\n")
            f.write(f"Created: {config_data['timestamp']}\n\n")
            
            for section, params in config_data.items():
                if isinstance(params, dict):
                    f.write(f"[{section.upper()}]\n")
                    for key, value in params.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
        
        return test_path
    
    def initialize_metrics_tracking(self) -> Dict:
        """
        Initialize metrics tracking structure
        
        Returns:
            Empty metrics history dictionary
        """
        return {
            'train': {
                'epochs': [],
                'total_loss': [],
                'mse_loss': [],
                'corr_loss': [],
                'speed_loss': [],
                'distance': []
            },
            'val': {
                'epochs': [],
                'total_loss': [],
                'mse_loss': [],
                'corr_loss': [],
                'speed_loss': [],
                'distance': []
            }
        }
    
    def save_epoch_metrics(self, test_path: str, epoch: int, 
                          train_metrics: Dict, val_metrics: Dict,
                          best_metrics: Dict, metrics_history: Dict,
                          training_info: Dict) -> None:
        """
        Save metrics for a specific epoch
        
        Args:
            test_path: Path to test directory
            epoch: Current epoch number
            train_metrics: Training metrics for this epoch
            val_metrics: Validation metrics for this epoch (can be None)
            best_metrics: Best metrics so far
            metrics_history: Complete metrics history
            training_info: Training information
        """
        # Save individual epoch metrics
        metrics_file = os.path.join(test_path, 'metrics', f'metrics_epoch_{epoch:08d}.json')
        current_metrics = {
            'epoch': epoch,
            'timestamp': datetime.datetime.now().isoformat(),
            'train': train_metrics,
            'best_so_far': best_metrics
        }
        
        if val_metrics:
            current_metrics['val'] = val_metrics
        
        with open(metrics_file, 'w') as f:
            json.dump(current_metrics, f, indent=4)
        
        # Save complete metrics history
        history_file = os.path.join(test_path, 'metrics', 'complete_metrics_history.json')
        complete_history = {
            'training_info': training_info,
            'metrics_history': metrics_history,
            'last_updated': datetime.datetime.now().isoformat(),
            'total_epochs_completed': len(metrics_history['train']['epochs'])
        }
        
        with open(history_file, 'w') as f:
            json.dump(complete_history, f, indent=4)
    
    def save_training_summary(self, test_path: str, test_number: int,
                            metrics_history: Dict, training_info: Dict,
                            current_epoch: int, train_metrics: Dict,
                            val_metrics: Optional[Dict] = None,
                            best_metrics: Optional[Dict] = None) -> None:
        """
        Save training summary
        
        Args:
            test_path: Path to test directory
            test_number: Test number
            metrics_history: Complete metrics history
            training_info: Training information
            current_epoch: Current epoch
            train_metrics: Current training metrics
            val_metrics: Current validation metrics (optional)
            best_metrics: Best metrics so far (optional)
        """
        summary_file = os.path.join(test_path, 'metrics', 'training_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"=== {self.script_name.upper()} Test {test_number} Training Summary ===\n")
            f.write(f"Last Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("[CURRENT EPOCH RESULTS]\n")
            f.write(f"Epoch: {current_epoch}\n")
            f.write(f"Train Loss: {train_metrics.get('total_loss', 'N/A'):.6f}\n")
            if 'mse_loss' in train_metrics:
                f.write(f"  MSE: {train_metrics['mse_loss']:.6f}, ")
                f.write(f"Corr: {train_metrics.get('corr_loss', 0):.6f}, ")
                f.write(f"Speed: {train_metrics.get('speed_loss', 0):.6f}\n")
            f.write(f"Train Distance: {train_metrics.get('distance', 'N/A'):.6f}\n")
            
            if val_metrics:
                f.write(f"Val Loss: {val_metrics.get('total_loss', 'N/A'):.6f}\n")
                if 'mse_loss' in val_metrics:
                    f.write(f"  MSE: {val_metrics['mse_loss']:.6f}, ")
                    f.write(f"Corr: {val_metrics.get('corr_loss', 0):.6f}, ")
                    f.write(f"Speed: {val_metrics.get('speed_loss', 0):.6f}\n")
                f.write(f"Val Distance: {val_metrics.get('distance', 'N/A'):.6f}\n")
            
            f.write("\n")
            
            if best_metrics:
                f.write("[BEST RESULTS SO FAR]\n")
                f.write(f"Best Val Loss: {best_metrics.get('val_loss_min', 'N/A'):.6f}\n")
                f.write(f"Best Val Distance: {best_metrics.get('val_dist_min', 'N/A'):.6f}\n\n")
            
            if len(metrics_history['train']['epochs']) > 1:
                f.write("[TRAINING PROGRESS]\n")
                f.write(f"Total Epochs: {len(metrics_history['train']['epochs'])}\n")
                f.write(f"Initial Train Loss: {metrics_history['train']['total_loss'][0]:.6f}\n")
                f.write(f"Current Train Loss: {metrics_history['train']['total_loss'][-1]:.6f}\n")
                improvement = metrics_history['train']['total_loss'][0] - metrics_history['train']['total_loss'][-1]
                f.write(f"Train Loss Improvement: {improvement:.6f}\n")
    
    def save_final_summary(self, test_path: str, test_number: int,
                          metrics_history: Dict, training_info: Dict,
                          loss_config: Optional[Dict] = None) -> str:
        """
        Save final training summary
        
        Args:
            test_path: Path to test directory
            test_number: Test number
            metrics_history: Complete metrics history
            training_info: Training information
            loss_config: Loss configuration
            
        Returns:
            Path to final summary file
        """
        final_summary_file = os.path.join(test_path, 'FINAL_TRAINING_SUMMARY.txt')
        
        # Update training info with final results
        training_info['end_time'] = datetime.datetime.now().isoformat()
        training_info['total_epochs'] = len(metrics_history['train']['epochs'])
        
        if metrics_history['val']['total_loss']:
            training_info['best_val_loss'] = min(metrics_history['val']['total_loss'])
            training_info['best_val_distance'] = min(metrics_history['val']['distance'])
        
        with open(final_summary_file, 'w') as f:
            f.write(f"=== FINAL {self.script_name.upper()} TRAINING SUMMARY - TEST {test_number} ===\n")
            f.write(f"Training completed: {training_info['end_time']}\n")
            
            if 'start_time' in training_info:
                start_time = datetime.datetime.fromisoformat(training_info['start_time'])
                end_time = datetime.datetime.fromisoformat(training_info['end_time'])
                f.write(f"Total training time: {end_time - start_time}\n\n")
            
            f.write("[CONFIGURATION]\n")
            if loss_config:
                f.write("Loss Configuration:\n")
                for key, value in loss_config.items():
                    f.write(f"  {key}: {value}\n")
            f.write(f"Device: {training_info.get('device', 'N/A')}\n")
            f.write(f"Model parameters: {training_info.get('model_parameters', 'N/A'):,}\n")
            f.write(f"Trainable parameters: {training_info.get('trainable_parameters', 'N/A'):,}\n\n")
            
            f.write("[FINAL RESULTS]\n")
            f.write(f"Total epochs: {training_info['total_epochs']}\n")
            if 'best_val_loss' in training_info:
                f.write(f"Best validation loss: {training_info['best_val_loss']:.6f}\n")
                f.write(f"Best validation distance: {training_info['best_val_distance']:.6f}\n")
            
            if len(metrics_history['train']['epochs']) > 0:
                f.write(f"Final train loss: {metrics_history['train']['total_loss'][-1]:.6f}\n")
                f.write(f"Final train distance: {metrics_history['train']['distance'][-1]:.6f}\n")
                
                if len(metrics_history['train']['epochs']) > 1:
                    improvement = metrics_history['train']['total_loss'][0] - metrics_history['train']['total_loss'][-1]
                    f.write(f"Total train loss improvement: {improvement:.6f}\n")
        
        return final_summary_file
    
    def print_test_header(self, test_number: int, test_path: str) -> None:
        """
        Print formatted test header
        
        Args:
            test_number: Test number
            test_path: Path to test directory
        """
        print(f"\n{'='*60}")
        print(f" STARTING {self.script_name.upper()} TEST {test_number}")
        print(f" Test Directory: {test_path}")
        print(f"⏰ Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
    
    def print_completion_summary(self, test_number: int, test_path: str,
                               metrics_history: Dict, final_summary_file: str) -> None:
        """
        Print training completion summary
        
        Args:
            test_number: Test number
            test_path: Path to test directory
            metrics_history: Complete metrics history
            final_summary_file: Path to final summary file
        """
        print(f"\n{'='*60}")
        print(f" {self.script_name.upper()} TRAINING COMPLETED FOR TEST {test_number}")
        print(f"⏰ Finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[DATA] Total epochs completed: {len(metrics_history['train']['epochs'])}")
        
        if metrics_history['val']['total_loss']:
            print(f"[TARGET] Best validation loss: {min(metrics_history['val']['total_loss']):.6f}")
            print(f" Best validation distance: {min(metrics_history['val']['distance']):.6f}")
        
        print(f" Results saved in: {test_path}")
        print(f" Final summary: {final_summary_file}")
        print(f"{'='*60}\n")
