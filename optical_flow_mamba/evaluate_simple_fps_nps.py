"""
Evaluation Script for Simplified FPS/NPS Model

Evaluate the trained model and compare with baseline performance.
"""

import os
import sys
import torch
import json
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from main utils
from utils.loader import Dataset
from utils.funs import pair_samples, read_calib_matrices, reference_image_points
from utils.transform import LabelTransform, PredictionTransform, PointTransform

# Import simplified FPS/NPS modules
from config import SimpleFPSNPSConfig
from network_simple_fps_nps import create_simple_fps_nps_model
from simple_losses import create_simple_loss_function


def load_trained_model(model_path, config, device):
    """Load trained model"""
    
    # Create model
    model_config = config.get_model_config()
    model = create_simple_fps_nps_model(model_config).to(device)
    
    # Load weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[OK] Model loaded from: {model_path}")
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    return model


def evaluate_model(model, test_loader, device, transform_label, 
                   transform_prediction, transform_into_points):
    """Evaluate model on test set"""
    
    model.eval()
    
    all_distances = []
    all_losses = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for frames, tforms, _, _ in test_loader:
            frames, tforms = frames.to(device), tforms.to(device)
            tforms_inv = torch.linalg.inv(tforms)
            frames = frames / 255.0
            
            # Transform labels
            labels = transform_label(tforms, tforms_inv)
            
            # Forward pass
            outputs = model(frames)
            preds = transform_prediction(outputs)
            
            # Transform to points
            preds_pts = transform_into_points(preds)
            labels_pts = transform_into_points(labels)
            
            # Calculate point distances
            distances = ((preds_pts - labels_pts) ** 2).sum(dim=2).sqrt()  # (B, N)
            mean_distances = distances.mean(dim=1)  # (B,)
            
            # Store results
            all_distances.extend(mean_distances.cpu().numpy())
            all_predictions.append(preds_pts.cpu().numpy())
            all_targets.append(labels_pts.cpu().numpy())
    
    return np.array(all_distances), all_predictions, all_targets


def compute_metrics(distances):
    """Compute evaluation metrics"""
    
    metrics = {
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'median_distance': np.median(distances),
        'min_distance': np.min(distances),
        'max_distance': np.max(distances),
        'percentile_95': np.percentile(distances, 95),
        'percentile_99': np.percentile(distances, 99),
        'success_rate_0_5mm': np.mean(distances < 0.5) * 100,
        'success_rate_1_0mm': np.mean(distances < 1.0) * 100,
        'success_rate_2_0mm': np.mean(distances < 2.0) * 100,
    }
    
    return metrics


def plot_results(distances, save_path):
    """Plot evaluation results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Distance histogram
    axes[0, 0].hist(distances, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(distances), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(distances):.3f}mm')
    axes[0, 0].axvline(np.median(distances), color='green', linestyle='--', 
                       label=f'Median: {np.median(distances):.3f}mm')
    axes[0, 0].set_xlabel('Point Distance (mm)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Point Distances')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_distances = np.sort(distances)
    cumulative = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances) * 100
    axes[0, 1].plot(sorted_distances, cumulative, linewidth=2)
    axes[0, 1].axvline(0.5, color='red', linestyle='--', alpha=0.7, label='0.5mm target')
    axes[0, 1].axvline(1.0, color='orange', linestyle='--', alpha=0.7, label='1.0mm')
    axes[0, 1].axvline(2.0, color='yellow', linestyle='--', alpha=0.7, label='2.0mm')
    axes[0, 1].set_xlabel('Point Distance (mm)')
    axes[0, 1].set_ylabel('Cumulative Percentage (%)')
    axes[0, 1].set_title('Cumulative Distribution of Point Distances')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot
    axes[1, 0].boxplot(distances, vert=True)
    axes[1, 0].set_ylabel('Point Distance (mm)')
    axes[1, 0].set_title('Box Plot of Point Distances')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Performance comparison
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    success_rates = [np.mean(distances < t) * 100 for t in thresholds]
    
    axes[1, 1].bar(range(len(thresholds)), success_rates, alpha=0.7)
    axes[1, 1].set_xticks(range(len(thresholds)))
    axes[1, 1].set_xticklabels([f'{t}mm' for t in thresholds])
    axes[1, 1].set_ylabel('Success Rate (%)')
    axes[1, 1].set_title('Success Rate at Different Thresholds')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add values on bars
    for i, rate in enumerate(success_rates):
        axes[1, 1].text(i, rate + 1, f'{rate:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_path, 'evaluation_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[DATA] Results plot saved to: {plot_path}")
    
    plt.show()


def main():
    """Main evaluation function"""
    
    print(" Starting Simplified FPS/NPS Model Evaluation")
    print("=" * 60)
    
    # Setup configuration
    config = SimpleFPSNPSConfig()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Setup data
    full_dataset = Dataset(
        data_path=config.DATA_PATH,
        num_samples=config.NUM_SAMPLES,
        sample_range=config.SAMPLE_RANGE
    )
    
    # Use test fold for evaluation
    dataset_folds = full_dataset.partition_by_ratio(
        ratios=[1] * 5,
        randomise=True
    )
    test_dataset = dataset_folds[4]  # Test fold
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.MINIBATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Setup transformations
    data_pairs = pair_samples(config.NUM_SAMPLES, config.NUM_PRED, 0).to(device)
    
    tform_calib_scale, tform_calib_R_T, tform_calib = read_calib_matrices(
        os.path.join(os.getcwd(), config.FILENAME_CALIB)
    )
    
    sample_frame = full_dataset[0][0]
    image_points = reference_image_points(sample_frame.shape[1:], 2).to(device)
    
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
    
    # Load trained model
    model_path = os.path.join(config.SAVE_PATH, 'saved_model', 'best_model.pth')
    model = load_trained_model(model_path, config, device)
    
    # Evaluate model
    print("\n Evaluating model...")
    distances, predictions, targets = evaluate_model(
        model, test_loader, device, transform_label, 
        transform_prediction, transform_into_points
    )
    
    # Compute metrics
    metrics = compute_metrics(distances)
    
    print("\n[DATA] Evaluation Results:")
    print("=" * 40)
    print(f"Mean Distance:     {metrics['mean_distance']:.4f} ± {metrics['std_distance']:.4f} mm")
    print(f"Median Distance:   {metrics['median_distance']:.4f} mm")
    print(f"Min Distance:      {metrics['min_distance']:.4f} mm")
    print(f"Max Distance:      {metrics['max_distance']:.4f} mm")
    print(f"95th Percentile:   {metrics['percentile_95']:.4f} mm")
    print(f"99th Percentile:   {metrics['percentile_99']:.4f} mm")
    print()
    print("Success Rates:")
    print(f"< 0.5mm:  {metrics['success_rate_0_5mm']:.1f}%")
    print(f"< 1.0mm:  {metrics['success_rate_1_0mm']:.1f}%")
    print(f"< 2.0mm:  {metrics['success_rate_2_0mm']:.1f}%")
    print("=" * 40)
    
    # Compare with targets
    target_distance = config.TARGET_POINT_DISTANCE
    if metrics['mean_distance'] < target_distance:
        print(f"[SUCCESS] SUCCESS! Mean distance {metrics['mean_distance']:.4f}mm < target {target_distance}mm")
    else:
        print(f"[ERROR] Target not met. Mean distance {metrics['mean_distance']:.4f}mm > target {target_distance}mm")
    
    # Save results
    results = {
        'metrics': metrics,
        'config': config.__dict__,
        'model_path': model_path,
        'evaluation_date': str(datetime.datetime.now())
    }
    
    results_path = os.path.join(config.SAVE_PATH, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n Results saved to: {results_path}")
    
    # Plot results
    plot_results(distances, config.SAVE_PATH)
    
    print("\n[OK] Evaluation completed!")


if __name__ == "__main__":
    main()
