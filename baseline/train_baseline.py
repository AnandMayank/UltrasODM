#!/usr/bin/env python3
"""
Training script for Baseline Model (Model A)
Optical Flow + EfficientNet

This script trains the baseline model for comparison with the contrastive frame grouping approach.
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from datetime import datetime
import argparse
from torch.utils.tensorboard import SummaryWriter

from utils.network_comparison import build_comparison_model
from utils.comparison_losses import build_loss_function
from utils.loader import Dataset
from utils.funs import load_config
from utils.metrics import cal_dist


class BaselineTrainer:
    """Trainer for Baseline Model A: Optical Flow + EfficientNet"""
    
    def __init__(self, config_path):
        self.config_path = config_path
        # load_config returns (args_dict, args_namespace)
        _, self.opt = load_config(config_path)
        
        # Setup device
        self.device = torch.device(f'cuda:{self.opt.GPU}' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create save directory
        os.makedirs(self.opt.SAVE_PATH, exist_ok=True)
        
        # Initialize model
        self.model = build_comparison_model(
            'baseline',
            in_frames=self.opt.IN_FRAMES,
            pred_dim=self.opt.PRED_DIM,
            input_channels=self.opt.INPUT_CHANNELS
        ).to(self.device)
        
        # Initialize loss function
        self.criterion = build_loss_function(
            'baseline',
            alpha_mse=self.opt.ALPHA_MSE,
            alpha_corr=self.opt.ALPHA_CORR,
            alpha_speed=self.opt.ALPHA_SPEED
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.opt.LEARNING_RATE,
            weight_decay=self.opt.WEIGHT_DECAY
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=self.opt.SCHEDULER_FACTOR, 
            patience=self.opt.SCHEDULER_PATIENCE, verbose=True
        )
        
        # Initialize data loaders
        self._setup_data_loaders()
        
        # Initialize logging
        if self.opt.USE_TENSORBOARD:
            self.writer = SummaryWriter(os.path.join(self.opt.SAVE_PATH, 'tensorboard'))
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        print(f"Baseline Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def _setup_data_loaders(self):
        """Setup training and validation data loaders"""
        # Load dataset
        dataset = Dataset(
            data_path=self.opt.DATA_PATH,
            num_samples=self.opt.IN_FRAMES,
            sample_range=self.opt.SAMPLE_RANGE
        )
        
        # Split dataset (80% train, 20% validation)
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.opt.BATCH_SIZE,
            shuffle=True,
            num_workers=self.opt.NUM_WORKERS,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.opt.BATCH_SIZE,
            shuffle=False,
            num_workers=self.opt.NUM_WORKERS,
            pin_memory=True
        )
        
        print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_corr = 0
        total_speed = 0
        num_batches = 0

        for batch_idx, (frames, tforms, _, _) in enumerate(self.train_loader):
            frames = frames.to(self.device).float() / 255.0  # Normalize to [0,1]
            tforms = tforms.to(self.device).float()

            # Extract 6DoF parameters from transformation matrices
            # This is a simplified extraction - you may need to adjust based on your data format
            targets = tforms.view(tforms.shape[0], -1)[:, :self.opt.PRED_DIM]

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(frames)

            # Compute loss
            loss, loss_dict = self.criterion(predictions, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss_dict['total_loss']
            total_mse += loss_dict['mse_loss']
            total_corr += loss_dict['correlation_loss']
            total_speed += loss_dict['speed_loss']
            num_batches += 1

            # Log progress
            if batch_idx % self.opt.FREQ_INFO == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss_dict["total_loss"]:.4f}, '
                      f'MSE: {loss_dict["mse_loss"]:.4f}, '
                      f'Corr: {loss_dict["correlation_loss"]:.4f}, '
                      f'Speed: {loss_dict["speed_loss"]:.4f}')

        # Average metrics
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_corr = total_corr / num_batches
        avg_speed = total_speed / num_batches

        return avg_loss, avg_mse, avg_corr, avg_speed
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_corr = 0
        total_speed = 0
        num_batches = 0

        with torch.no_grad():
            for frames, tforms, _, _ in self.val_loader:
                frames = frames.to(self.device).float() / 255.0
                tforms = tforms.to(self.device).float()

                # Extract targets
                targets = tforms.view(tforms.shape[0], -1)[:, :self.opt.PRED_DIM]

                # Forward pass
                predictions = self.model(frames)

                # Compute loss
                _, loss_dict = self.criterion(predictions, targets)

                # Accumulate metrics
                total_loss += loss_dict['total_loss']
                total_mse += loss_dict['mse_loss']
                total_corr += loss_dict['correlation_loss']
                total_speed += loss_dict['speed_loss']
                num_batches += 1

        # Average metrics
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_corr = total_corr / num_batches
        avg_speed = total_speed / num_batches

        return avg_loss, avg_mse, avg_corr, avg_speed
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.opt
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.opt.SAVE_PATH, f'{self.opt.MODEL_SAVE_NAME}_epoch_{epoch:04d}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.opt.SAVE_PATH, f'{self.opt.MODEL_SAVE_NAME}_best.pth')
            torch.save(checkpoint, best_path)
            print(f"[OK] New best model saved with validation loss: {self.best_val_loss:.4f}")
    
    def train(self):
        """Main training loop"""
        print("Starting Baseline Model training...")
        print(f"Model: Optical Flow + EfficientNet")
        print(f"Losses: MSE (α={self.opt.ALPHA_MSE}) + Correlation (α={self.opt.ALPHA_CORR}) + Speed (α={self.opt.ALPHA_SPEED})")
        print("=" * 60)

        for epoch in range(self.start_epoch, self.opt.NUM_EPOCHS):
            start_time = time.time()

            # Train
            train_loss, train_mse, train_corr, train_speed = self.train_epoch(epoch)

            # Validate
            val_loss, val_mse, val_corr, val_speed = self.validate_epoch(epoch)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Save losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Log to tensorboard
            if self.opt.USE_TENSORBOARD:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Validation', val_loss, epoch)
                self.writer.add_scalar('MSE/Train', train_mse, epoch)
                self.writer.add_scalar('MSE/Validation', val_mse, epoch)
                self.writer.add_scalar('Correlation/Train', train_corr, epoch)
                self.writer.add_scalar('Correlation/Validation', val_corr, epoch)
                self.writer.add_scalar('Speed/Train', train_speed, epoch)
                self.writer.add_scalar('Speed/Validation', val_speed, epoch)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save checkpoint
            if epoch % self.opt.FREQ_SAVE == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch:3d}/{self.opt.NUM_EPOCHS} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Time: {epoch_time:.1f}s')
            
            # Early stopping
            if hasattr(self.opt, 'EARLY_STOPPING') and self.opt.EARLY_STOPPING:
                if len(self.val_losses) > self.opt.EARLY_STOPPING_PATIENCE:
                    recent_losses = self.val_losses[-self.opt.EARLY_STOPPING_PATIENCE:]
                    if all(loss >= self.best_val_loss for loss in recent_losses):
                        print(f"Early stopping triggered after {epoch} epochs")
                        break
        
        print("Training completed!")
        if self.opt.USE_TENSORBOARD:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Baseline Model')
    parser.add_argument('--config', type=str, default='configs/baseline_config.txt',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Initialize and start training
    trainer = BaselineTrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
