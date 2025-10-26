"""
Unified data loader for 2D and 3D medical image segmentation.

This module provides functionality to:
- Load 2D slices or 3D volumes from Nifti files
- Normalize and preprocess images
- Apply data augmentation (2D: rotation, flip; 3D: downsampling)
- Create train/validation/test splits
- Support both 2D (keras slices) and 3D (semantic volumes) data

Usage:
    2D: create_data_loaders(data_dir, batch_size=32, mode='2d')
    3D: create_data_loaders(data_dir, batch_size=1, mode='3d', downsample_factor=2)
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
from scipy.ndimage import zoom


class MedicalImageDataset(Dataset):
    """
    PyTorch Dataset for 2D medical image slices with optional augmentation.
    
    Args:
        image_paths: List of paths to image files
        label_paths: List of paths to label/segmentation files
        augment: Whether to apply data augmentation (default: False)
        normalize: Whether to normalize images to 0-1 range (default: True)
        standardize: Whether to standardize images (zero mean, unit variance) (default: False)
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


# ========== 3D VOLUMETRIC DATASET ==========

class VolumetricDataset(Dataset):
    """
    PyTorch Dataset for 3D volumetric medical images.
    
    Args:
        volume_paths: List of paths to volume files
        label_paths: List of paths to label/segmentation files
        downsample_factor: Factor to downsample volumes (default: 2, reduces by 2x in each dimension)
        normalize: Whether to normalize images (default: True)
    """
    
    def __init__(self, volume_paths, label_paths, downsample_factor=2, normalize=True):
        """Initialize the volumetric dataset."""
        assert len(volume_paths) == len(label_paths), "Number of volumes and labels must match"
        
        self.volume_paths = volume_paths
        self.label_paths = label_paths
        self.downsample_factor = downsample_factor
        self.normalize = normalize
        
        # Cache loaded volumes to speed up training
        self.volume_cache = {}
        self.label_cache = {}
    
    def __len__(self):
        """Return the number of volumes in the dataset."""
        return len(self.volume_paths)
    
    def _downsample_volume(self, volume):
        """
        Downsample 3D volume using scipy.ndimage.zoom.
        
        Args:
            volume: 3D numpy array
            
        Returns:
            Downsampled 3D numpy array
        """
        if self.downsample_factor == 1:
            return volume
        
        # Calculate zoom factors (inverse of downsample factor)
        zoom_factor = 1.0 / self.downsample_factor
        return zoom(volume, zoom_factor, order=1)  # order=1 for bilinear interpolation
    
    def _normalize_volume(self, volume):
        """
        Normalize volume to [0, 1] range using min-max normalization.
        
        Args:
            volume: 3D numpy array
            
        Returns:
            Normalized 3D numpy array
        """
        vol_min = volume.min()
        vol_max = volume.max()
        
        if vol_max - vol_min == 0:
            return np.zeros_like(volume, dtype=np.float32)
        
        normalized = (volume - vol_min) / (vol_max - vol_min)
        return normalized.astype(np.float32)
    
    def __getitem__(self, idx):
        """
        Get a volume and its corresponding label.
        
        Args:
            idx: Index of the volume
            
        Returns:
            Tuple of (volume tensor, label tensor) each with shape (C, D, H, W) or (D, H, W)
        """
        
        volume_path = self.volume_paths[idx]
        label_path = self.label_paths[idx]
        
        # Load from cache or disk
        if idx not in self.volume_cache:
            # Load volume
            img = nib.load(str(volume_path))
            volume = img.get_fdata()
            
            # Downsample
            volume = self._downsample_volume(volume)
            
            # Normalize
            if self.normalize:
                volume = self._normalize_volume(volume)
            
            # Add channel dimension: (D, H, W) -> (1, D, H, W)
            volume = np.expand_dims(volume, axis=0).astype(np.float32)
            
            self.volume_cache[idx] = torch.from_numpy(volume)
        
        # Load label
        if idx not in self.label_cache:
            # Load label
            lbl_img = nib.load(str(label_path))
            label = lbl_img.get_fdata()
            
            # Downsample label with nearest neighbor
            label = zoom(label, 1.0 / self.downsample_factor, order=0)
            
            # Keep as (D, H, W) - no channel dimension for labels
            label = label.astype(np.int64)
            
            self.label_cache[idx] = torch.from_numpy(label)
        
        return self.volume_cache[idx], self.label_cache[idx]


def load_semantic_data(data_dir, downsample_factor=2):
    """
    Load 3D semantic segmentation data from directory structure.
    
    Expected structure:
    semantic_MRs/
    ├── volume_001.nii.gz
    ├── volume_002.nii.gz
    └── ...
    
    semantic_labels_only/
    ├── volume_001.nii.gz
    ├── volume_002.nii.gz
    └── ...
    
    Args:
        data_dir: Path to parent directory containing semantic_MRs and semantic_labels_only
        downsample_factor: Downsampling factor for 3D volumes
        
    Returns:
        Tuple of (volume_paths, label_paths) lists sorted alphabetically
    """
    
    data_dir = Path(data_dir)
    
    volume_dir = data_dir / "semantic_MRs"
    label_dir = data_dir / "semantic_labels_only"
    
    if not volume_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            f"Required directories not found:\n"
            f"  Volumes: {volume_dir}\n"
            f"  Labels: {label_dir}"
        )
    
    # Find all nifti files
    volume_files = sorted(
        list(volume_dir.glob("*.nii")) + list(volume_dir.glob("*.nii.gz"))
    )
    label_files = sorted(
        list(label_dir.glob("*.nii")) + list(label_dir.glob("*.nii.gz"))
    )
    
    if len(volume_files) == 0:
        raise FileNotFoundError(f"No NIfTI files found in {volume_dir}")
    
    if len(label_files) == 0:
        raise FileNotFoundError(f"No NIfTI files found in {label_dir}")
    
    print(f"Found {len(volume_files)} volumes and {len(label_files)} labels")
    
    return volume_files, label_files


def create_3d_data_loaders(data_dir, batch_size=1, num_workers=0, 
                          downsample_factor=2, train_split=0.6, val_split=0.2):
    """
    Create train/validation/test DataLoaders for 3D semantic segmentation.
    
    Args:
        data_dir: Path to data directory with semantic_MRs and semantic_labels_only
        batch_size: Batch size for DataLoaders (default: 1 for memory efficiency)
        num_workers: Number of worker threads (default: 0)
        downsample_factor: Downsampling factor for 3D volumes (default: 2)
        train_split: Fraction for training set (default: 0.6)
        val_split: Fraction for validation set (default: 0.2)
        
    Returns:
        Dictionary with train_loader, val_loader, test_loader, and dataset info
    """
    
    # Load file paths
    volume_files, label_files = load_semantic_data(data_dir, downsample_factor)
    
    # Create dataset
    dataset = VolumetricDataset(
        volume_files, 
        label_files,
        downsample_factor=downsample_factor,
        normalize=True
    )
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
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
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'total_samples': total_size
    }


# ========== UNIFIED DATA LOADER (2D/3D) ==========

def create_data_loaders(data_dir, batch_size=32, num_workers=4, augment_train=True, 
                       mode='2d', downsample_factor=2, train_split=0.6, val_split=0.2):
    """
    Unified function to create DataLoaders for either 2D or 3D data.
    
    Args:
        data_dir: Path to data directory (keras_slices_data for 2D or root for 3D)
        batch_size: Batch size (default: 32 for 2D, recommend 1 for 3D)
        num_workers: Number of workers (default: 4, recommend 0 for 3D)
        augment_train: Whether to augment training data (default: True, only for 2D)
        mode: '2d' for 2D slices or '3d' for 3D volumes (default: '2d')
        downsample_factor: Downsampling factor for 3D (default: 2)
        train_split: Training split fraction (default: 0.6)
        val_split: Validation split fraction (default: 0.2)
        
    Returns:
        For 2D: (train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)
        For 3D: Dictionary with loaders and datasets
    """
    
    if mode.lower() == '3d':
        return create_3d_data_loaders(
            data_dir, 
            batch_size=batch_size, 
            num_workers=num_workers,
            downsample_factor=downsample_factor,
            train_split=train_split,
            val_split=val_split
        )
    else:
        # 2D mode
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
