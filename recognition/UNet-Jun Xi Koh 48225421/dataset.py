"""
Data loader for 2D medical image slices with preprocessing and augmentation.

This module provides functionality to:
- Load 2D slices from Nifti files
- Normalize and preprocess images
- Apply data augmentation (rotation, flip, etc.)
- Create train/validation/test splits
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import nibabel as nib
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.transforms import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip
from torch.nn.functional import interpolate


class MedicalImageDataset(Dataset):
    """
    PyTorch Dataset for 2D medical image slices.
    
    Args:
        image_paths: List of paths to image files
        label_paths: List of paths to label/segmentation files
        augment: Whether to apply data augmentation (default: False)
        normalize: Whether to normalize images to 0-1 range (default: True)
        standardize: Whether to standardize images (zero mean, unit variance) (default: True)
        target_size: Target size (H, W) to resize all images to (default: (256, 256))
    """
    
    def __init__(self, image_paths, label_paths, augment=False, normalize=True, standardize=False, target_size=(256, 256)):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.augment = augment
        self.normalize = normalize
        self.standardize = standardize
        self.target_size = target_size
        self.images = []
        self.labels = []
        
        # Load images and labels into memory for faster training
        self._load_data()
        
        # Define augmentation transforms
        self.augmentation_transforms = transforms.Compose([
            RandomRotation(degrees=15),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ]) if augment else None
    
    def _load_data(self):
        """Load all images and labels from disk."""
        print("Loading dataset...")
        
        for img_path, label_path in tqdm(zip(self.image_paths, self.label_paths)):
            # Load image
            img_nifti = nib.load(img_path)
            img = img_nifti.get_fdata(caching='unchanged').astype(np.float32)
            
            # Handle 3D images by taking first slice if needed
            if len(img.shape) == 3:
                img = img[:, :, 0]
            
            # Load label
            label_nifti = nib.load(label_path)
            label = label_nifti.get_fdata(caching='unchanged').astype(np.int64)
            
            # Handle 3D labels by taking first slice if needed
            if len(label.shape) == 3:
                label = label[:, :, 0]
            
            # Resize to target size to ensure all images have the same dimensions
            if img.shape != self.target_size:
                # Convert to tensor, resize, and convert back
                img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
                img_resized = interpolate(img_tensor, size=self.target_size, mode='bilinear', align_corners=False)
                img = img_resized.squeeze(0).squeeze(0).numpy()
            
            if label.shape != self.target_size:
                # Use nearest neighbor for labels to preserve class values
                label_tensor = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
                label_resized = interpolate(label_tensor, size=self.target_size, mode='nearest')
                label = label_resized.squeeze(0).squeeze(0).numpy().astype(np.int64)
            
            # Preprocess image
            if self.normalize:
                # Normalize to 0-1 range
                img_min = img.min()
                img_max = img.max()
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min)
                else:
                    img = np.zeros_like(img)
            
            if self.standardize:
                # Standardize to zero mean and unit variance
                img_mean = img.mean()
                img_std = img.std()
                if img_std > 0:
                    img = (img - img_mean) / img_std
            
            self.images.append(img)
            self.labels.append(label)
        
        print(f"Loaded {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        image = self.images[idx].copy()
        label = self.labels[idx].copy()
        
        # Add channel dimension for images
        image = np.expand_dims(image, axis=0)  # (1, H, W)
        
        # Apply augmentation if training
        if self.augment and self.augmentation_transforms:
            # Convert to tensor for transforms
            image_tensor = torch.from_numpy(image).float()
            label_tensor = torch.from_numpy(label).long().unsqueeze(0)  # (1, H, W)
            
            # Apply same transform to both image and label
            seed = np.random.randint(0, 2**31)
            torch.manual_seed(seed)
            image_tensor = self.augmentation_transforms(image_tensor)
            torch.manual_seed(seed)
            label_tensor = self.augmentation_transforms(label_tensor)
            
            image = image_tensor.numpy()
            label = label_tensor.squeeze(0).numpy()
        
        # Convert to tensors
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        
        return image, label


def load_keras_slices(data_dir):
    """
    Load 2D Keras slices dataset from directory structure.
    
    Expected directory structure:
    - keras_slices_data/
      - keras_slices_train/  (images)
      - keras_slices_seg_train/  (segmentations)
      - keras_slices_validate/  (images)
      - keras_slices_seg_validate/  (segmentations)
      - keras_slices_test/  (images)
      - keras_slices_seg_test/  (segmentations)
    
    Args:
        data_dir: Path to keras_slices_data directory
        
    Returns:
        Tuple of (train_images, train_labels, val_images, val_labels, test_images, test_labels)
    """
    
    data_dir = Path(data_dir)
    
    # Training set
    train_img_dir = data_dir / "keras_slices_train"
    train_label_dir = data_dir / "keras_slices_seg_train"
    
    train_images = sorted([str(f) for f in train_img_dir.glob("*.nii*")])
    train_labels = sorted([str(f) for f in train_label_dir.glob("*.nii*")])
    
    # Validation set
    val_img_dir = data_dir / "keras_slices_validate"
    val_label_dir = data_dir / "keras_slices_seg_validate"
    
    val_images = sorted([str(f) for f in val_img_dir.glob("*.nii*")])
    val_labels = sorted([str(f) for f in val_label_dir.glob("*.nii*")])
    
    # Test set
    test_img_dir = data_dir / "keras_slices_test"
    test_label_dir = data_dir / "keras_slices_seg_test"
    
    test_images = sorted([str(f) for f in test_img_dir.glob("*.nii*")])
    test_labels = sorted([str(f) for f in test_label_dir.glob("*.nii*")])
    
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    print(f"Test samples: {len(test_images)}")
    
    return (train_images, train_labels, 
            val_images, val_labels, 
            test_images, test_labels)


def create_data_loaders(data_dir, batch_size=32, num_workers=4, augment_train=True):
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        data_dir: Path to keras_slices_data directory
        batch_size: Batch size for DataLoaders (default: 32)
        num_workers: Number of workers for DataLoader (default: 4)
        augment_train: Whether to apply augmentation to training data (default: True)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Load image and label paths
    train_images, train_labels, val_images, val_labels, test_images, test_labels = \
        load_keras_slices(data_dir)
    
    # Create datasets
    train_dataset = MedicalImageDataset(
        train_images, train_labels,
        augment=augment_train,
        normalize=True,
        standardize=False
    )
    
    val_dataset = MedicalImageDataset(
        val_images, val_labels,
        augment=False,
        normalize=True,
        standardize=False
    )
    
    test_dataset = MedicalImageDataset(
        test_images, test_labels,
        augment=False,
        normalize=True,
        standardize=False
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
