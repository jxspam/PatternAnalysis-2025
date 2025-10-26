"""
Unified inference script for 2D and 3D Improved UNet segmentation.

This script demonstrates:
- Loading a trained model (2D or 3D)
- Making predictions on single images/volumes
- Visualizing results
- Computing Dice scores

Usage:
    2D: python predict.py --mode 2d --checkpoint ./checkpoints/best_model_epoch_1.pt --image_path path/to/image.nii.gz
    3D: python predict.py --mode 3d --checkpoint ./checkpoints_3d/best_model_epoch_1.pt --volume_path path/to/volume.nii.gz
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
from scipy.ndimage import zoom

from modules import ImprovedUNet2D, ImprovedUNet3D, dice_coefficient, dice_coefficient_3d


def load_model(checkpoint_path, mode='2d', num_classes=2, device=None):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pt file, not full checkpoint)
        mode: '2d' or '3d'
        num_classes: Number of classes (default: 2)
        device: Device to load model on (default: cuda if available, else cpu)
        
    Returns:
        Loaded model on specified device
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    if mode.lower() == '3d':
        model = ImprovedUNet3D(in_channels=1, num_classes=num_classes, filters=32, dropout_rate=0.5)
    else:
        model = ImprovedUNet2D(in_channels=1, num_classes=num_classes, filters=64, dropout_rate=0.5)
    
    # Load checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"Loaded {mode.upper()} model from {checkpoint_path}")
    return model


def predict_single_image(model, image_path, device=None):
    """
    Predict segmentation for a single 2D image.
    
    Args:
        model: Trained model
        image_path: Path to NIfTI image file
        device: Device to use (default: cuda if available, else cpu)
        
    Returns:
        Tuple of (prediction, normalized_image)
            prediction: Predicted class indices (H, W)
            normalized_image: Normalized input image (H, W)
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load image
    img_nifti = nib.load(str(image_path))
    image = img_nifti.get_fdata().astype(np.float32)
    
    # Handle 3D by taking first slice
    if len(image.shape) == 3:
        image = image[:, :, 0]
    
    # Normalize
    img_min = image.min()
    img_max = image.max()
    if img_max > img_min:
        image = (image - img_min) / (img_max - img_min)
    
    # Convert to tensor and add batch/channel dims
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # (H, W)
    
    return prediction, image


def predict_single_volume(model, volume_path, downsample_factor=2, device=None):
    """
    Predict segmentation for a single 3D volume.
    
    Args:
        model: Trained model
        volume_path: Path to NIfTI volume file
        downsample_factor: Downsampling factor used during training (default: 2)
        device: Device to use (default: cuda if available, else cpu)
        
    Returns:
        Tuple of (prediction, downsampled_volume, original_volume)
            prediction: Predicted class indices (D, H, W) at downsampled resolution
            downsampled_volume: Input volume at downsampled resolution
            original_volume: Original volume at native resolution
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load volume
    vol_nifti = nib.load(str(volume_path))
    volume = vol_nifti.get_fdata().astype(np.float32)
    
    # Downsample
    if downsample_factor > 1:
        zoom_factor = 1.0 / downsample_factor
        volume_downsampled = zoom(volume, zoom_factor, order=1)
    else:
        volume_downsampled = volume.copy()
    
    # Normalize
    vol_min = volume_downsampled.min()
    vol_max = volume_downsampled.max()
    if vol_max > vol_min:
        volume_downsampled = (volume_downsampled - vol_min) / (vol_max - vol_min)
    
    # Convert to tensor and add batch/channel dims
    volume_tensor = torch.from_numpy(volume_downsampled).float().unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, D, H, W)
    
    # Predict
    with torch.no_grad():
        output = model(volume_tensor)
        prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # (D, H, W)
    
    return prediction, volume_downsampled, volume


def evaluate_predictions(predictions, ground_truth, num_classes=2):
    """
    Evaluate predictions against ground truth.
    
    Args:
        predictions: Predicted class indices (array)
        ground_truth: Ground truth labels (array)
        num_classes: Number of classes
        
    Returns:
        Dictionary with Dice scores for each class and average
    """
    
    pred_tensor = torch.from_numpy(predictions).long()
    gt_tensor = torch.from_numpy(ground_truth).long()
    
    # For 2D, add batch dimension
    if pred_tensor.dim() == 2:
        pred_tensor = pred_tensor.unsqueeze(0)
        gt_tensor = gt_tensor.unsqueeze(0)
    
    # Choose correct dice function based on dimensionality
    if pred_tensor.dim() == 3:  # 2D: (B, H, W)
        # Add class dimension for dice_coefficient
        pred_one_hot = torch.zeros(pred_tensor.shape[0], num_classes, *pred_tensor.shape[1:])
        for c in range(num_classes):
            pred_one_hot[:, c] = (pred_tensor == c)
        return dice_coefficient(pred_one_hot, gt_tensor, num_classes=num_classes)
    else:  # 3D: (B, D, H, W) or (D, H, W)
        if pred_tensor.dim() == 4:
            pred_tensor = pred_tensor.squeeze(0)
            gt_tensor = gt_tensor.squeeze(0)
        return dice_coefficient_3d(
            torch.nn.functional.one_hot(pred_tensor, num_classes=num_classes).permute(3, 0, 1, 2).unsqueeze(0),
            gt_tensor.unsqueeze(0),
            num_classes=num_classes
        )


def visualize_2d_prediction(image, prediction, ground_truth=None, save_path=None):
    """
    Visualize 2D prediction results.
    
    Args:
        image: Input image (H, W)
        prediction: Predicted segmentation (H, W)
        ground_truth: Ground truth segmentation (H, W), optional
        save_path: Path to save visualization, optional
    """
    
    num_plots = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))
    if num_plots == 2:
        axes = [axes[0], axes[1]]
    
    # Input image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(prediction, cmap='jet')
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # Ground truth
    if ground_truth is not None:
        axes[2].imshow(ground_truth, cmap='jet')
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    plt.show()


def visualize_3d_prediction_slices(volume, prediction, ground_truth=None, num_slices=5, save_path=None):
    """
    Visualize 3D prediction results by showing middle slices.
    
    Args:
        volume: Input 3D volume (D, H, W)
        prediction: Predicted 3D segmentation (D, H, W)
        ground_truth: Ground truth 3D segmentation (D, H, W), optional
        num_slices: Number of slices to show (default: 5)
        save_path: Path to save visualization, optional
    """
    
    # Get middle slices
    depth = volume.shape[0]
    slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
    
    num_plots = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(num_slices, num_plots, figsize=(12, 4*num_slices))
    
    for row, slice_idx in enumerate(slice_indices):
        # Input slice
        axes[row, 0].imshow(volume[slice_idx], cmap='gray')
        axes[row, 0].set_title(f'Input Slice {slice_idx}')
        axes[row, 0].axis('off')
        
        # Prediction slice
        axes[row, 1].imshow(prediction[slice_idx], cmap='jet')
        axes[row, 1].set_title(f'Prediction Slice {slice_idx}')
        axes[row, 1].axis('off')
        
        # Ground truth slice
        if ground_truth is not None:
            axes[row, 2].imshow(ground_truth[slice_idx], cmap='jet')
            axes[row, 2].set_title(f'GT Slice {slice_idx}')
            axes[row, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict with Improved UNet model')
    parser.add_argument('--mode', type=str, default='2d', choices=['2d', '3d'],
                        help='Prediction mode: 2d or 3d')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image_path', type=str,
                        help='Path to 2D image (for 2d mode)')
    parser.add_argument('--volume_path', type=str,
                        help='Path to 3D volume (for 3d mode)')
    parser.add_argument('--downsample_factor', type=int, default=2,
                        help='Downsampling factor for 3D')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Load model
    model = load_model(args.checkpoint, mode=args.mode, device=device)
    
    if args.mode == '2d':
        if not args.image_path:
            raise ValueError("--image_path required for 2d mode")
        
        # Predict
        pred, img = predict_single_image(model, args.image_path, device=device)
        
        # Visualize
        visualize_2d_prediction(img, pred, save_path="prediction_2d.png")
        
    else:  # 3d mode
        if not args.volume_path:
            raise ValueError("--volume_path required for 3d mode")
        
        # Predict
        pred, vol_ds, vol = predict_single_volume(model, args.volume_path, 
                                                  downsample_factor=args.downsample_factor,
                                                  device=device)
        
        # Visualize
        visualize_3d_prediction_slices(vol_ds, pred, num_slices=5, save_path="prediction_3d.png")
        
        print(f"Prediction shape: {pred.shape}")
        print(f"Classes: {np.unique(pred)}")
