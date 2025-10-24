"""
Visualization utilities for UNet segmentation training and inference.

This module provides functions to:
- Plot training metrics (loss, Dice coefficient) across epochs
- Visualize segmentation predictions on test samples
- Generate comprehensive reports with multiple visualizations
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json


def plot_dice_loss_curves(history: Dict, output_dir: Path, figsize=(15, 10)):
    """
    Plot comprehensive training curves for Dice loss and metrics.
    
    Args:
        history: Dictionary with training history containing:
                - train_loss: List of training losses per epoch
                - val_loss: List of validation losses per epoch
                - val_dice: List of validation Dice scores per epoch
        output_dir: Directory to save plots
        figsize: Figure size (default: (15, 10))
        
    Returns:
        Path to saved figure
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Training vs Validation Loss
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2.5, marker='o')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2.5, marker='s')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Dice Loss', fontsize=12)
    axes[0, 0].set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(range(1, len(epochs) + 1, max(1, len(epochs)//10)))
    
    # 2. Validation Dice Score
    val_dice_avg = [d['avg'] for d in history['val_dice']]
    axes[0, 1].plot(epochs, val_dice_avg, 'g-', linewidth=2.5, marker='o', label='Avg Dice')
    axes[0, 1].axhline(y=0.75, color='red', linestyle='--', linewidth=2, label='Target (0.75)')
    axes[0, 1].fill_between(epochs, val_dice_avg, alpha=0.3, color='green')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Dice Coefficient', fontsize=12)
    axes[0, 1].set_title('Validation Dice Score per Epoch', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].set_xticks(range(1, len(epochs) + 1, max(1, len(epochs)//10)))
    
    # 3. Per-class Dice scores
    if 'class_0' in history['val_dice'][0] and 'class_1' in history['val_dice'][0]:
        class_0_dice = [d['class_0'] for d in history['val_dice']]
        class_1_dice = [d['class_1'] for d in history['val_dice']]
        
        axes[1, 0].plot(epochs, class_0_dice, 'b-', linewidth=2.5, marker='o', label='Background (Class 0)')
        axes[1, 0].plot(epochs, class_1_dice, 'r-', linewidth=2.5, marker='s', label='Tissue (Class 1)')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Dice Coefficient', fontsize=12)
        axes[1, 0].set_title('Per-Class Dice Scores', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].set_xticks(range(1, len(epochs) + 1, max(1, len(epochs)//10)))
    
    # 4. Loss improvement over time
    min_val_loss = min(history['val_loss'])
    loss_improvement = [history['val_loss'][0] - l for l in history['val_loss']]
    axes[1, 1].bar(epochs, loss_improvement, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 1].axhline(y=0, color='red', linestyle='-', linewidth=1)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss Improvement from Epoch 1', fontsize=12)
    axes[1, 1].set_title('Validation Loss Improvement', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_xticks(range(1, len(epochs) + 1, max(1, len(epochs)//10)))
    
    plt.tight_layout()
    
    # Add summary text
    best_epoch = history['best_epoch']
    best_dice = history['best_val_dice']
    fig.suptitle(f'Training Summary - Best Epoch: {best_epoch} (Dice: {best_dice:.4f})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plot_path = output_dir / 'dice_loss_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Dice loss curves to {plot_path}")
    plt.close(fig)
    
    return plot_path


def plot_test_predictions(images: np.ndarray, predictions: np.ndarray, labels: np.ndarray, 
                          dice_scores: List[Dict], output_dir: Path, 
                          sample_indices: List[int] = None, figsize_per_sample=(15, 5)):
    """
    Plot test set predictions with ground truth and Dice scores.
    
    Args:
        images: Array of input images (N, H, W)
        predictions: Array of predictions (N, H, W)
        labels: Array of ground truth labels (N, H, W)
        dice_scores: List of Dice score dictionaries for each sample
        output_dir: Directory to save plots
        sample_indices: Specific sample indices to visualize (if None, shows all)
        figsize_per_sample: Figure size per sample (default: (15, 5))
        
    Returns:
        List of paths to saved figures
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_samples = len(images)
    saved_paths = []
    
    if sample_indices is None:
        sample_indices = range(num_samples)
    
    for idx in sample_indices:
        if idx >= num_samples:
            break
            
        fig, axes = plt.subplots(1, 3, figsize=figsize_per_sample)
        
        image = images[idx]
        prediction = predictions[idx]
        label = labels[idx]
        dice = dice_scores[idx]
        
        # Input image
        im0 = axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Prediction
        im1 = axes[1].imshow(prediction, cmap='viridis', vmin=0, vmax=1)
        dice_str = f"Avg Dice: {dice['avg']:.4f}\nClass 0: {dice.get('class_0', 0):.4f}\nClass 1: {dice.get('class_1', 0):.4f}"
        axes[1].set_title(f'Prediction\n{dice_str}', fontsize=11, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Ground truth
        im2 = axes[2].imshow(label, cmap='viridis', vmin=0, vmax=1)
        axes[2].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Test Sample {idx:03d}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        fig_path = output_dir / f'test_sample_{idx:03d}.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        saved_paths.append(fig_path)
        plt.close(fig)
    
    print(f"✓ Saved {len(saved_paths)} test prediction visualizations to {output_dir}")
    return saved_paths


def plot_test_grid(images: np.ndarray, predictions: np.ndarray, labels: np.ndarray, 
                   dice_scores: List[Dict], output_dir: Path, grid_size=(3, 4)):
    """
    Plot a grid of test samples for quick comparison.
    
    Args:
        images: Array of input images (N, H, W)
        predictions: Array of predictions (N, H, W)
        labels: Array of ground truth labels (N, H, W)
        dice_scores: List of Dice score dictionaries
        output_dir: Directory to save plot
        grid_size: Grid dimensions (rows, cols) (default: (3, 4))
        
    Returns:
        Path to saved figure
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rows, cols = grid_size
    num_samples = min(rows * cols, len(images))
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx in range(num_samples):
        prediction = predictions[idx]
        dice = dice_scores[idx]['avg']
        
        axes[idx].imshow(prediction, cmap='viridis')
        axes[idx].set_title(f'Sample {idx} (Dice: {dice:.3f})', fontsize=10)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Test Set Predictions Grid ({num_samples} samples)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig_path = output_dir / 'test_predictions_grid.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved test predictions grid to {fig_path}")
    plt.close(fig)
    
    return fig_path


def plot_dice_distribution(dice_scores: List[Dict], output_dir: Path, figsize=(14, 5)):
    """
    Plot distribution of Dice scores across test set.
    
    Args:
        dice_scores: List of Dice score dictionaries
        output_dir: Directory to save plot
        figsize: Figure size (default: (14, 5))
        
    Returns:
        Path to saved figure
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract values
    avg_dice = [d['avg'] for d in dice_scores]
    class_0_dice = [d.get('class_0', 0) for d in dice_scores]
    class_1_dice = [d.get('class_1', 0) for d in dice_scores]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Average Dice histogram
    axes[0].hist(avg_dice, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(avg_dice), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(avg_dice):.4f}')
    axes[0].axvline(np.median(avg_dice), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(avg_dice):.4f}')
    axes[0].set_xlabel('Dice Coefficient', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Average Dice Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Class 0 Dice histogram
    axes[1].hist(class_0_dice, bins=20, color='orange', edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(class_0_dice), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(class_0_dice):.4f}')
    axes[1].set_xlabel('Dice Coefficient', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Class 0 (Background) Dice', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Class 1 Dice histogram
    axes[2].hist(class_1_dice, bins=20, color='purple', edgecolor='black', alpha=0.7)
    axes[2].axvline(np.mean(class_1_dice), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(class_1_dice):.4f}')
    axes[2].set_xlabel('Dice Coefficient', fontsize=11)
    axes[2].set_ylabel('Frequency', fontsize=11)
    axes[2].set_title('Class 1 (Tissue) Dice', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    fig_path = output_dir / 'dice_distribution.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved Dice distribution plot to {fig_path}")
    plt.close(fig)
    
    return fig_path


def create_visualization_summary(history: Dict, test_images: np.ndarray, 
                                test_predictions: np.ndarray, test_labels: np.ndarray,
                                test_dice_scores: List[Dict], output_dir: Path):
    """
    Create comprehensive visualization summary with all plots.
    
    Args:
        history: Training history dictionary
        test_images: Array of test images
        test_predictions: Array of test predictions
        test_labels: Array of test labels
        test_dice_scores: List of test Dice scores
        output_dir: Directory to save all visualizations
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION SUMMARY")
    print("="*60)
    
    # 1. Plot training curves
    plot_dice_loss_curves(history, output_dir)
    
    # 2. Plot Dice distribution
    plot_dice_distribution(test_dice_scores, output_dir)
    
    # 3. Plot test predictions grid
    plot_test_grid(test_images, test_predictions, test_labels, test_dice_scores, output_dir)
    
    # 4. Plot individual test samples (first 10)
    num_to_plot = min(10, len(test_images))
    plot_test_predictions(test_images, test_predictions, test_labels, 
                         test_dice_scores, output_dir, 
                         sample_indices=list(range(num_to_plot)))
    
    # 5. Create summary statistics file
    summary_stats = {
        'training': {
            'num_epochs': len(history['train_loss']),
            'best_epoch': history['best_epoch'],
            'best_val_dice': float(history['best_val_dice']),
            'final_train_loss': float(history['train_loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1]),
            'min_train_loss': float(min(history['train_loss'])),
            'min_val_loss': float(min(history['val_loss'])),
        },
        'test_results': {
            'num_test_samples': len(test_images),
            'avg_dice': float(np.mean([d['avg'] for d in test_dice_scores])),
            'std_dice': float(np.std([d['avg'] for d in test_dice_scores])),
            'min_dice': float(min([d['avg'] for d in test_dice_scores])),
            'max_dice': float(max([d['avg'] for d in test_dice_scores])),
        }
    }
    
    summary_path = output_dir / 'visualization_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=4)
    print(f"✓ Saved visualization summary to {summary_path}")
    
    print("="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    
    return output_dir
