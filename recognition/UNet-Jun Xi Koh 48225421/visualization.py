"""
Visualization module for 3D UNet segmentation results.

Provides functions to:
1. Plot training curves (Dice loss across epochs)
2. Generate comparison images (predicted vs ground truth)
3. Create grid visualizations of multiple samples
4. Generate 10 test samples per PNG image with prostate comparisons
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import json
from scipy.ndimage import zoom

from modules import ImprovedUNet3D, dice_coefficient_3d
from dataset import create_data_loaders




def plot_training_curves(history_path, output_dir='./results', save_name='training_curves_3d.png'):
    """
    Plot training curves from training history JSON for 3D training.
    
    Creates two plots:
    1. Train and Validation Loss across epochs
    2. Validation Dice coefficient (average and per-class) across epochs
    
    Args:
        history_path: Path to training_history_3d.json
        output_dir: Directory to save plots
        save_name: Name of output PNG file
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Extract data
    epochs = range(1, len(history['train_loss']) + 1)
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    
    # Extract Dice scores (avg across classes)
    val_dice_avg = [d['avg'] for d in history['val_dice']]
    val_dice_class0 = [d['class_0'] for d in history['val_dice']]
    val_dice_class1 = [d['class_1'] for d in history['val_dice']]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss curves
    axes[0].plot(epochs, train_loss, 'o-', label='Train Loss', linewidth=2, markersize=6)
    axes[0].plot(epochs, val_loss, 's-', label='Validation Loss', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Dice Loss (1 - Dice)', fontsize=12, fontweight='bold')
    axes[0].set_title('Training and Validation Loss Across Epochs', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(1, len(epochs) + 1, max(1, len(epochs) // 10)))
    
    # Plot 2: Dice coefficient curves
    axes[1].plot(epochs, val_dice_avg, 'o-', label='Avg Dice', linewidth=2, markersize=6, color='green')
    axes[1].plot(epochs, val_dice_class0, 's--', label='Background Dice', linewidth=2, markersize=5, color='blue')
    axes[1].plot(epochs, val_dice_class1, '^--', label='Prostate Dice', linewidth=2, markersize=5, color='red')
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Dice Coefficient', fontsize=12, fontweight='bold')
    axes[1].set_title('Validation Dice Coefficient Across Epochs', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])
    axes[1].set_xticks(range(1, len(epochs) + 1, max(1, len(epochs) // 10)))
    axes[1].axhline(y=0.7, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='Threshold (0.7)')
    
    # Add best epoch marker
    best_epoch = history['best_epoch']
    best_val_dice = history['best_val_dice']
    axes[1].axvline(x=best_epoch, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[1].text(best_epoch, best_val_dice + 0.02, f'Best: Epoch {best_epoch}', 
                ha='center', fontsize=10, fontweight='bold', color='green')
    
    plt.tight_layout()
    output_path = output_dir / save_name
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved training curves to {output_path}")
    plt.close()


def visualize_3d_predictions_grid_10_per_image(model, test_loader, device, num_batches=None,
                                               output_dir='./results', downsample_factor=2):
    """
    Generate PNG files with 10 test samples per image showing prostate segmentation.
    
    Each PNG contains 10 samples arranged in 2 rows, with multiple slices per sample:
    - Column 1: Input volume slice (middle slice)
    - Column 2: Predicted prostate segmentation
    - Column 3: Ground truth segmentation
    
    Args:
        model: Trained 3D UNet model
        test_loader: DataLoader for test set
        device: Device to use (cuda/cpu)
        num_batches: Number of batches to process (None = all batches)
        output_dir: Directory to save plots
        downsample_factor: Downsampling factor used during training
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Collect all test samples
    all_volumes = []
    all_predictions = []
    all_labels = []
    
    batch_count = 0
    with torch.no_grad():
        for batch_data in test_loader:
            if num_batches is not None and batch_count >= num_batches:
                break
            
            volumes, labels = batch_data
            volumes = volumes.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(volumes)
            predictions = torch.argmax(outputs, dim=1)
            
            # Collect results
            for i in range(volumes.shape[0]):
                all_volumes.append(volumes[i, 0].cpu().numpy())
                all_predictions.append(predictions[i].cpu().numpy())
                all_labels.append(labels[i].cpu().numpy())
            
            batch_count += 1
    
    # Create figures with 10 samples each
    num_samples = len(all_volumes)
    samples_per_image = 10
    num_images = (num_samples + samples_per_image - 1) // samples_per_image
    
    for img_idx in range(num_images):
        start_idx = img_idx * samples_per_image
        end_idx = min(start_idx + samples_per_image, num_samples)
        num_samples_in_image = end_idx - start_idx
        
        # Determine grid size (e.g., 5 samples per row, 2 rows)
        rows = (num_samples_in_image + 4) // 5
        cols = min(num_samples_in_image, 5)
        
        fig = plt.figure(figsize=(cols * 3, rows * 3))
        gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.35, wspace=0.3)
        
        for local_idx, sample_idx in enumerate(range(start_idx, end_idx)):
            volume = all_volumes[sample_idx]
            prediction = all_predictions[sample_idx]
            ground_truth = all_labels[sample_idx]
            
            depth = volume.shape[0]
            middle_slice = depth // 2
            
            row = local_idx // cols
            col = local_idx % cols
            ax = fig.add_subplot(gs[row, col])
            
            # Create a simple 2D visualization: input + predicted outline
            input_img = volume[middle_slice]
            pred_mask = (prediction[middle_slice] == 1).astype(float)
            gt_mask = (ground_truth[middle_slice] == 1).astype(float)
            
            # Display input in grayscale
            ax.imshow(input_img, cmap='gray', alpha=0.7)
            # Overlay predicted prostate in red
            ax.imshow(np.ma.masked_where(pred_mask == 0, pred_mask), cmap='Reds', alpha=0.5, label='Predicted')
            # Overlay GT prostate in blue contour
            ax.contour(gt_mask, levels=[0.5], colors='blue', linewidths=2, label='Ground Truth')
            
            ax.set_title(f'Sample {sample_idx + 1}', fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for local_idx in range(num_samples_in_image, rows * cols):
            row = local_idx // cols
            col = local_idx % cols
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')
        
        plt.suptitle(f'3D UNet Test Predictions - Samples {start_idx + 1} to {end_idx}\n' + 
                    '(Red: Predicted Prostate, Blue: Ground Truth Contour)',
                    fontsize=12, fontweight='bold', y=0.98)
        
        output_path = output_dir / f'test_predictions_batch_{img_idx + 1:02d}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved batch {img_idx + 1} ({num_samples_in_image} samples) to {output_path}")
        plt.close()


def visualize_3d_detailed_comparison(model, test_loader, device, num_samples=10,
                                     output_dir='./results', downsample_factor=2):
    """
    Generate detailed comparison visualizations for first N test samples.
    
    Each sample shows 5 slices with 5 columns:
    1. Input volume
    2. Predicted prostate mask
    3. Ground truth mask
    4. Prediction overlay on input
    5. Ground truth overlay on input
    
    Args:
        model: Trained 3D UNet model
        test_loader: DataLoader for test set
        device: Device to use (cuda/cpu)
        num_samples: Number of test samples to visualize in detail
        output_dir: Directory to save plots
        downsample_factor: Downsampling factor used during training
    """
    
    output_dir = Path(output_dir)
    predictions_dir = output_dir / 'predictions_detailed'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    samples_collected = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            if samples_collected >= num_samples:
                break
            
            volumes, labels = batch_data
            volumes = volumes.to(device)
            labels = labels.to(device)
            
            batch_size = volumes.shape[0]
            
            # Forward pass
            outputs = model(volumes)
            predictions = torch.argmax(outputs, dim=1)
            
            for i in range(batch_size):
                if samples_collected >= num_samples:
                    break
                
                volume = volumes[i, 0].cpu().numpy()
                prediction = predictions[i].cpu().numpy()
                ground_truth = labels[i].cpu().numpy()
                
                # Create figure with multiple slices
                depth = volume.shape[0]
                num_views = 5
                num_slices_to_show = 5
                
                fig = plt.figure(figsize=(15, 10))
                
                # Select middle slices
                slice_indices = np.linspace(0, depth - 1, num_slices_to_show, dtype=int)
                
                col = 0
                for slice_idx in slice_indices:
                    # Input volume
                    ax = plt.subplot(num_slices_to_show, num_views, col * num_views + 1)
                    ax.imshow(volume[slice_idx], cmap='gray')
                    if col == 0:
                        ax.set_ylabel(f'Slice {slice_idx}', fontweight='bold')
                    if col == 0:
                        ax.set_title('Input Volume', fontweight='bold')
                    ax.axis('off')
                    
                    # Predicted prostate mask
                    ax = plt.subplot(num_slices_to_show, num_views, col * num_views + 2)
                    ax.imshow(prediction[slice_idx], cmap='jet', vmin=0, vmax=1)
                    if col == 0:
                        ax.set_title('Predicted', fontweight='bold')
                    ax.axis('off')
                    
                    # Ground truth mask
                    ax = plt.subplot(num_slices_to_show, num_views, col * num_views + 3)
                    ax.imshow(ground_truth[slice_idx], cmap='jet', vmin=0, vmax=1)
                    if col == 0:
                        ax.set_title('Ground Truth', fontweight='bold')
                    ax.axis('off')
                    
                    # Overlay: prediction on input
                    ax = plt.subplot(num_slices_to_show, num_views, col * num_views + 4)
                    ax.imshow(volume[slice_idx], cmap='gray')
                    ax.imshow(np.ma.masked_where(prediction[slice_idx] == 0, prediction[slice_idx]),
                             cmap='Reds', alpha=0.5)
                    if col == 0:
                        ax.set_title('Pred Overlay', fontweight='bold')
                    ax.axis('off')
                    
                    # Overlay: ground truth on input
                    ax = plt.subplot(num_slices_to_show, num_views, col * num_views + 5)
                    ax.imshow(volume[slice_idx], cmap='gray')
                    ax.imshow(np.ma.masked_where(ground_truth[slice_idx] == 0, ground_truth[slice_idx]),
                             cmap='Blues', alpha=0.5)
                    if col == 0:
                        ax.set_title('GT Overlay', fontweight='bold')
                    ax.axis('off')
                    
                    col += 1
                
                plt.suptitle(f'Sample {samples_collected + 1:03d} - 3D Prostate Segmentation Analysis',
                            fontsize=12, fontweight='bold', y=0.995)
                plt.tight_layout()
                
                output_path = predictions_dir / f'sample_{samples_collected + 1:03d}.png'
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"✓ Saved detailed sample {samples_collected + 1} to {output_path}")
                plt.close()
                
                samples_collected += 1


def generate_all_3d_visualizations(checkpoint_dir, data_dir, device=None, 
                                   num_detailed_samples=10, downsample_factor=2):
    """
    Generate all 3D visualizations (training curves + test predictions).
    
    Creates:
    1. training_curves_3d.png: Loss and Dice curves across epochs
    2. test_predictions_batch_*.png: 10 samples per image with overlays
    3. predictions_detailed/sample_*.png: Detailed 5-slice comparisons for first N samples
    
    Args:
        checkpoint_dir: Directory containing checkpoints and training history
        data_dir: Directory containing test data
        device: Device to use (cuda/cpu)
        num_detailed_samples: Number of samples for detailed analysis
        downsample_factor: Downsampling factor used during training
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_dir = Path(checkpoint_dir)
    
    # Find best model
    history_path = checkpoint_dir / 'training_history_3d.json'
    
    if not history_path.exists():
        print(f"ERROR: Training history not found at {history_path}")
        return
    
    # Load history to get best epoch
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    best_epoch = history['best_epoch']
    best_model_path = checkpoint_dir / f'best_model_epoch_{best_epoch}.pt'
    
    if not best_model_path.exists():
        print(f"ERROR: Best model not found at {best_model_path}")
        return
    
    # Load model
    print("Loading trained model...")
    model = ImprovedUNet3D(in_channels=1, num_classes=2, filters=32, dropout_rate=0.5).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"✓ Loaded best model from epoch {best_epoch}")
    
    # Load test data
    print("Loading test data...")
    data_loaders = create_data_loaders(data_dir, batch_size=1, num_workers=0, mode='3d',
                                      downsample_factor=downsample_factor)
    test_loader = data_loaders['test_loader']
    
    # Create results directory
    results_dir = Path('./results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Plot training curves
    print("\nGenerating training curves...")
    plot_training_curves(history_path, output_dir=results_dir)
    
    # 2. Generate test predictions grid (10 per image)
    print("\nGenerating test predictions with 10 samples per image...")
    visualize_3d_predictions_grid_10_per_image(model, test_loader, device,
                                               output_dir=results_dir,
                                               downsample_factor=downsample_factor)
    
    # 3. Generate detailed per-sample visualizations
    print(f"\nGenerating detailed per-sample visualizations ({num_detailed_samples} samples)...")
    visualize_3d_detailed_comparison(model, test_loader, device, 
                                     num_samples=num_detailed_samples,
                                     output_dir=results_dir, 
                                     downsample_factor=downsample_factor)
    
    print(f"\n✓ All 3D visualizations saved to {results_dir}")
    print(f"  - training_curves_3d.png: Training loss and Dice curves")
    print(f"  - test_predictions_batch_*.png: Test samples (10 per image)")
    print(f"  - predictions_detailed/sample_*.png: Detailed multi-slice analysis")
    
    return results_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate 3D UNet training and prediction visualizations')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_3d',
                        help='Directory containing model checkpoints and training history')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to HipMRI_Study_open directory')
    parser.add_argument('--num_detailed', type=int, default=10,
                        help='Number of samples for detailed visualization')
    parser.add_argument('--downsample_factor', type=int, default=2,
                        help='Downsampling factor used during training')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    generate_all_3d_visualizations(args.checkpoint_dir, args.data_dir, device=device,
                                  num_detailed_samples=args.num_detailed,
                                  downsample_factor=args.downsample_factor)