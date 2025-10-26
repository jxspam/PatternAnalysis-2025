"""
Inference script for 3D Improved UNet segmentation.

This script demonstrates:
- Loading a trained 3D model
- Making predictions on volumetric data
- Computing Dice scores for 3D predictions
- Visualizing 3D predictions as cross-sections
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
from tqdm import tqdm

from modules import ImprovedUNet3D, dice_coefficient_3d
from dataset3d import VolumetricDataset


def load_model_3d(checkpoint_path, num_classes=2, device=None):
    """
    Load a trained 3D model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_classes: Number of classes (default: 2)
        device: Device to load on (cuda or cpu)
        
    Returns:
        Loaded model on specified device
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ImprovedUNet3D(in_channels=1, num_classes=num_classes, filters=32)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


def predict_single_volume(model, volume_path, downsample_factor=2, device=None):
    """
    Make prediction on a single 3D volume.
    
    Args:
        model: Trained 3D UNet model
        volume_path: Path to NIfTI volume file
        downsample_factor: Downsampling factor used in training
        device: Device to run on
        
    Returns:
        Tuple of (prediction, downsampled_volume, original_volume)
        - prediction: (D, H, W) class indices
        - downsampled_volume: (D, H, W) normalized intensity
        - original_volume: (D, H, W) original intensity
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load volume
    img = nib.load(str(volume_path))
    volume = img.get_fdata()
    original_volume = volume.copy()
    
    # Create temporary dataset to handle preprocessing
    dataset = VolumetricDataset(
        [volume_path], [volume_path],  # Dummy label path
        downsample_factor=downsample_factor,
        normalize=True
    )
    
    # Get preprocessed volume
    processed_volume, _ = dataset[0]
    
    # Add batch dimension: (1, D, H, W) -> (1, 1, D, H, W)
    input_volume = processed_volume.unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        prediction = model(input_volume)
    
    # Convert to class indices
    prediction = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()
    
    # Get downsampled volume for visualization
    downsampled_volume = processed_volume.squeeze(0).numpy()
    
    return prediction, downsampled_volume, original_volume


def evaluate_3d_predictions(model, test_dataset, device=None):
    """
    Evaluate model on a set of 3D volumes.
    
    Args:
        model: Trained 3D model
        test_dataset: Test dataset with volumes and labels
        device: Device to run on
        
    Returns:
        Dictionary with per-sample and overall Dice scores
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    all_dice_scores = {f'class_{c}': [] for c in [0, 1]}
    all_dice_scores['avg'] = []
    
    print("\nEvaluating 3D predictions...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
            volume, label = test_dataset[idx]
            
            # Add batch dimension
            volume = volume.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)
            
            # Predict
            prediction = model(volume)
            
            # Calculate Dice
            dice = dice_coefficient_3d(prediction, label, num_classes=2)
            
            for key in all_dice_scores.keys():
                all_dice_scores[key].append(dice[key])
    
    # Calculate statistics
    results = {}
    for key in all_dice_scores.keys():
        scores = all_dice_scores[key]
        results[key] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'scores': scores
        }
    
    return results


def visualize_3d_prediction(volume, prediction, label=None, slice_idx=None, output_path=None):
    """
    Visualize 3D predictions as orthogonal cross-sections.
    
    Args:
        volume: 3D volume (D, H, W)
        prediction: 3D prediction (D, H, W)
        label: Optional ground truth label (D, H, W)
        slice_idx: Slice indices to visualize (default: center slices)
        output_path: Path to save visualization
    """
    
    D, H, W = volume.shape
    
    if slice_idx is None:
        slice_idx = {
            'depth': D // 2,
            'height': H // 2,
            'width': W // 2
        }
    
    # Determine number of subplots
    num_cols = 3 if label is not None else 2
    num_rows = 3  # Axial, sagittal, coronal
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 12))
    
    # Axial (D-H plane)
    d = slice_idx['depth']
    axes[0, 0].imshow(volume[d, :, :], cmap='gray')
    axes[0, 0].set_title(f'Axial (D={d}) - Input')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(prediction[d, :, :], cmap='nipy_spectral', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Axial (D={d}) - Prediction')
    axes[0, 1].axis('off')
    
    if label is not None:
        axes[0, 2].imshow(label[d, :, :], cmap='nipy_spectral', vmin=0, vmax=1)
        axes[0, 2].set_title(f'Axial (D={d}) - Ground Truth')
        axes[0, 2].axis('off')
    
    # Sagittal (D-W plane)
    w = slice_idx['width']
    axes[1, 0].imshow(volume[:, :, w], cmap='gray')
    axes[1, 0].set_title(f'Sagittal (W={w}) - Input')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(prediction[:, :, w], cmap='nipy_spectral', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Sagittal (W={w}) - Prediction')
    axes[1, 1].axis('off')
    
    if label is not None:
        axes[1, 2].imshow(label[:, :, w], cmap='nipy_spectral', vmin=0, vmax=1)
        axes[1, 2].set_title(f'Sagittal (W={w}) - Ground Truth')
        axes[1, 2].axis('off')
    
    # Coronal (H-W plane)
    h = slice_idx['height']
    axes[2, 0].imshow(volume[:, h, :], cmap='gray')
    axes[2, 0].set_title(f'Coronal (H={h}) - Input')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(prediction[:, h, :], cmap='nipy_spectral', vmin=0, vmax=1)
    axes[2, 1].set_title(f'Coronal (H={h}) - Prediction')
    axes[2, 1].axis('off')
    
    if label is not None:
        axes[2, 2].imshow(label[:, h, :], cmap='nipy_spectral', vmin=0, vmax=1)
        axes[2, 2].set_title(f'Coronal (H={h}) - Ground Truth')
        axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    
    return fig


def visualize_3d_stats(results, output_path=None):
    """
    Visualize Dice score statistics for 3D predictions.
    
    Args:
        results: Results dictionary from evaluate_3d_predictions()
        output_path: Path to save visualization
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Dice score distribution for each class
    classes = ['class_0', 'class_1', 'avg']
    colors = ['blue', 'red', 'green']
    
    for idx, (cls, color) in enumerate(zip(classes, colors)):
        scores = results[cls]['scores']
        axes[0].hist(scores, bins=10, alpha=0.6, label=cls, color=color)
    
    axes[0].set_xlabel('Dice Coefficient')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Dice Score Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Summary statistics
    means = [results[cls]['mean'] for cls in classes]
    stds = [results[cls]['std'] for cls in classes]
    
    x_pos = np.arange(len(classes))
    axes[1].bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(classes)
    axes[1].set_ylabel('Dice Coefficient')
    axes[1].set_title('Mean Dice Score ± Std Dev')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        axes[1].text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved statistics plot to {output_path}")
    
    return fig


if __name__ == "__main__":
    # Example usage
    checkpoint_path = "./checkpoints_3d/best_model_epoch_1.pt"
    volume_path = "./HipMRI_Study_open/semantic_MRs/volume_001.nii.gz"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    if Path(checkpoint_path).exists():
        model = load_model_3d(checkpoint_path, device=device)
        
        # Make prediction
        if Path(volume_path).exists():
            prediction, downsampled_vol, original_vol = predict_single_volume(
                model, volume_path, device=device
            )
            
            print(f"Prediction shape: {prediction.shape}")
            print(f"Prediction classes: {np.unique(prediction)}")
            
            # Visualize
            output_dir = Path("./prediction_results_3d")
            output_dir.mkdir(exist_ok=True)
            
            visualize_3d_prediction(
                downsampled_vol, prediction,
                output_path=str(output_dir / "sample_prediction.png")
            )
