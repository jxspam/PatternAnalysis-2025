"""
3D Data loader for volumetric medical image segmentation.

This module provides functionality to:
- Load 3D volumes from Nifti files
- Downsample volumes for memory efficiency
- Normalize and preprocess 3D images
- Create train/validation/test splits for 3D data
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom


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
            Tuple of (volume tensor, label tensor) each with shape (C, D, H, W)
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
