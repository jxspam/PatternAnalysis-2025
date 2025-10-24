"""
Inference script for Improved UNet segmentation.

This script demonstrates:
- Loading a trained model
- Making predictions on single images
- Visualizing results
- Computing Dice scores
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
from tqdm import tqdm

from modules import ImprovedUNet2D, dice_coefficient
from dataset import MedicalImageDataset
from visualization import plot_test_predictions, plot_dice_distribution, plot_test_grid


def load_model(checkpoint_path, num_classes=2, device=None):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_classes: Number of classes (default: 2)
        device: Device to load model on (default: cuda if available, else cpu)
        
    Returns:
        Loaded model on specified device
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = ImprovedUNet2D(in_channels=1, num_classes=num_classes, filters=64, dropout_rate=0.5)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    return model


def predict_single_image(model, image_path, device=None, normalize=True):
    """
    Make prediction on a single image.
    
    Args:
        model: Trained model
        image_path: Path to image file (Nifti format)
        device: Device to run inference on
        normalize: Whether to normalize image (default: True)
        
    Returns:
        Tuple of (prediction, image)
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load image
    img_nifti = nib.load(image_path)
    img = img_nifti.get_fdata(caching='unchanged').astype(np.float32)
    
    # Handle 3D images
    if len(img.shape) == 3:
        img = img[:, :, 0]
    
    # Normalize
    if normalize:
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
    
    # Add batch and channel dimensions
    img_tensor = torch.from_numpy(img).float()
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    img_tensor = img_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(img_tensor)
    
    # Convert to class indices
    prediction = torch.argmax(prediction, dim=1).cpu().numpy()
    
    return prediction[0], img


def evaluate_predictions(model, image_paths, label_paths, device=None):
    """
    Evaluate model predictions on multiple images.
    
    Args:
        model: Trained model
        image_paths: List of image paths
        label_paths: List of label paths
        device: Device to run inference on
        
    Returns:
        Dictionary with evaluation metrics
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_dice_scores = []
    
    print("Evaluating predictions...")
    for img_path, label_path in tqdm(zip(image_paths, label_paths)):
        # Make prediction
        prediction, img = predict_single_image(model, img_path, device=device)
        
        # Load label
        label_nifti = nib.load(label_path)
        label = label_nifti.get_fdata(caching='unchanged').astype(np.int64)
        if len(label.shape) == 3:
            label = label[:, :, 0]
        
        # Calculate Dice
        pred_tensor = torch.from_numpy(prediction).long().unsqueeze(0).unsqueeze(0)
        label_tensor = torch.from_numpy(label).long().unsqueeze(0)
        
        # Expand dimensions to match expected shape
        pred_tensor = pred_tensor.squeeze(1)  # (1, H, W)
        pred_probs = torch.zeros((1, 2, pred_tensor.shape[1], pred_tensor.shape[2]))
        for c in range(2):
            pred_probs[:, c] = (pred_tensor == c).float()
        
        dice_scores = dice_coefficient(pred_probs, label_tensor, num_classes=2)
        all_dice_scores.append(dice_scores)
    
    # Calculate average metrics
    avg_dice = {
        'class_0': np.mean([d['class_0'] for d in all_dice_scores]),
        'class_1': np.mean([d['class_1'] for d in all_dice_scores]),
        'avg': np.mean([d['avg'] for d in all_dice_scores])
    }
    
    return avg_dice


def visualize_prediction(image, prediction, label=None, figsize=(15, 5)):
    """
    Visualize image, prediction, and optionally ground truth label.
    
    Args:
        image: Input image (H, W)
        prediction: Model prediction (H, W)
        label: Ground truth label (H, W) (optional)
        figsize: Figure size (default: (15, 5))
    """
    
    if label is not None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Prediction
        axes[1].imshow(prediction, cmap='viridis')
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        # Ground truth
        axes[2].imshow(label, cmap='viridis')
        axes[2].set_title('Ground Truth Label')
        axes[2].axis('off')
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Prediction
        axes[1].imshow(prediction, cmap='viridis')
        axes[1].set_title('Prediction')
        axes[1].axis('off')
    
    plt.tight_layout()
    return fig


def main_example(checkpoint_path, data_dir, num_samples=5):
    """
    Example usage of the prediction script with comprehensive visualizations.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        data_dir: Path to test data directory
        num_samples: Number of samples to visualize (default: 5)
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(checkpoint_path, device=device)
    
    # Load test data paths
    test_dir = Path(data_dir)
    test_img_dir = test_dir / "keras_slices_test"
    test_label_dir = test_dir / "keras_slices_seg_test"
    
    test_images = sorted([str(f) for f in test_img_dir.glob("*.nii*")])
    test_labels = sorted([str(f) for f in test_label_dir.glob("*.nii*")])
    
    print(f"Found {len(test_images)} test images")
    
    # Create output directory
    output_dir = Path("./prediction_results")
    output_dir.mkdir(exist_ok=True)
    
    # Process all test data and collect results
    all_images = []
    all_predictions = []
    all_labels = []
    all_dice_scores = []
    
    print("\nProcessing all test samples...")
    for i in tqdm(range(len(test_images))):
        img_path = test_images[i]
        label_path = test_labels[i]
        
        # Make prediction
        prediction, image = predict_single_image(model, img_path, device=device)
        
        # Load label and resize if needed
        label_nifti = nib.load(label_path)
        label = label_nifti.get_fdata(caching='unchanged').astype(np.int64)
        if len(label.shape) == 3:
            label = label[:, :, 0]
        
        # Resize label to match prediction if needed
        if label.shape != prediction.shape:
            from torch.nn.functional import interpolate
            label_tensor = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float()
            label_resized = interpolate(label_tensor, size=prediction.shape, mode='nearest')
            label = label_resized.squeeze(0).squeeze(0).numpy().astype(np.int64)
        
        # Calculate Dice
        pred_tensor = torch.from_numpy(prediction).long().unsqueeze(0)
        pred_probs = torch.zeros((1, 2, pred_tensor.shape[1], pred_tensor.shape[2]))
        for c in range(2):
            pred_probs[:, c] = (pred_tensor == c).float()
        
        label_tensor = torch.from_numpy(label).long().unsqueeze(0)
        dice_scores = dice_coefficient(pred_probs, label_tensor, num_classes=2)
        
        # Store results
        all_images.append(image)
        all_predictions.append(prediction)
        all_labels.append(label)
        all_dice_scores.append(dice_scores)
    
    # Convert to numpy arrays
    all_images = np.array(all_images)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    print(f"\nProcessed {len(all_images)} test samples")
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING TEST VISUALIZATIONS")
    print("="*60)
    
    # 1. Plot test predictions grid
    plot_test_grid(all_images, all_predictions, all_labels, all_dice_scores, output_dir)
    
    # 2. Plot individual test samples
    num_to_plot = min(num_samples, len(all_images))
    plot_test_predictions(all_images, all_predictions, all_labels, all_dice_scores, 
                         output_dir, sample_indices=list(range(num_to_plot)))
    
    # 3. Plot Dice score distribution
    plot_dice_distribution(all_dice_scores, output_dir)
    
    # 4. Print summary statistics
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    avg_dice_scores = [d['avg'] for d in all_dice_scores]
    class_0_scores = [d.get('class_0', 0) for d in all_dice_scores]
    class_1_scores = [d.get('class_1', 0) for d in all_dice_scores]
    
    print(f"\nAverage Dice Score: {np.mean(avg_dice_scores):.4f} ± {np.std(avg_dice_scores):.4f}")
    print(f"  - Min: {np.min(avg_dice_scores):.4f}, Max: {np.max(avg_dice_scores):.4f}")
    print(f"  - Median: {np.median(avg_dice_scores):.4f}")
    
    print(f"\nClass 0 (Background) Dice: {np.mean(class_0_scores):.4f} ± {np.std(class_0_scores):.4f}")
    print(f"Class 1 (Tissue) Dice: {np.mean(class_1_scores):.4f} ± {np.std(class_1_scores):.4f}")
    
    print(f"\nAll visualizations saved to: {output_dir.absolute()}")
    print("="*60)
    
    return {
        'average_dice': float(np.mean(avg_dice_scores)),
        'std_dice': float(np.std(avg_dice_scores)),
        'min_dice': float(np.min(avg_dice_scores)),
        'max_dice': float(np.max(avg_dice_scores)),
        'class_0_dice': float(np.mean(class_0_scores)),
        'class_1_dice': float(np.mean(class_1_scores)),
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Inference with Improved UNet')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    metrics = main_example(args.checkpoint, args.data_dir, num_samples=args.num_samples)
