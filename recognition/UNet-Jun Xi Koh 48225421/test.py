"""
test.py - Test script for 2D and 3D Improved UNet on HipMRI Study

This script:
1. Validates dataset paths (Rangpur or local)
2. Trains the Improved UNet model (2D or 3D)
3. Evaluates on test set with Dice coefficient
4. Saves results and visualizations

For Rangpur (GPU): Uses /home/groups/comp3710/HipMRI_Study_open
For Local: Uses ./HipMRI_Study_open if Rangpur path not available

Usage:
    python test.py                    # Run 3D training with 50 epochs
    python test.py --mode 2d          # Run 2D training
    python test.py --mode 3d --epochs 50   # Run 3D with custom epochs
"""

from pathlib import Path
import sys
import torch
import nibabel as nib
import argparse

from train import train
from predict import load_model, evaluate_predictions
from dataset import load_keras_slices

# ============================================================================
# DEFAULT CONFIGURATION (Can be overridden by command-line arguments)
# ============================================================================

# 3D Training (recommended for better results - 50 epochs)
DEFAULT_MODE = '3d'
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 1
DEFAULT_DOWNSAMPLE_FACTOR = 2
DEFAULT_EARLY_STOPPING_PATIENCE = 5

# 2D Training (alternative - uses fewer resources)
# Uncomment below to default to 2D instead:
# DEFAULT_MODE = '2d'
# DEFAULT_EPOCHS = 50
# DEFAULT_BATCH_SIZE = 32

NUM_WORKERS = 0

def check_and_set_data_path():
    """
    Check for dataset availability on Rangpur or fallback to local.
    
    Returns:
        Path to keras_slices_data directory
    """
    
    # Try Rangpur path first
    rangpur_base = Path("/home/groups/comp3710/HipMRI_Study_open")
    rangpur_slices = rangpur_base / "keras_slices_data"
    
    if rangpur_base.exists() and rangpur_base.is_dir():
        if rangpur_slices.exists() and rangpur_slices.is_dir():
            print(f"✓ Using Rangpur dataset: {rangpur_slices}")
            return rangpur_slices
        else:
            print(f"WARNING: Rangpur base exists but slices not found: {rangpur_slices}")
    
    # Fallback to local path
    local_slices = Path("./HipMRI_Study_open/keras_slices_data")
    if local_slices.exists() and local_slices.is_dir():
        print(f"✓ Using local dataset: {local_slices}")
        return local_slices
    
    # Try alternative local path (relative to script)
    script_dir = Path(__file__).parent
    alt_local_slices = script_dir / "HipMRI_Study_open" / "keras_slices_data"
    if alt_local_slices.exists() and alt_local_slices.is_dir():
        print(f"✓ Using local dataset: {alt_local_slices}")
        return alt_local_slices
    
    print("ERROR: Could not find dataset paths:")
    print(f"  Rangpur: {rangpur_slices}")
    print(f"  Local: {local_slices}")
    print(f"  Alt Local: {alt_local_slices}")
    sys.exit(2)


def check_and_set_data_path_3d():
    """
    Check for 3D dataset availability on Rangpur or fallback to local.
    
    Returns:
        Path to semantic_MRs directory (volumes with labels)
    """
    
    # Try Rangpur path first
    rangpur_base = Path("/home/groups/comp3710/HipMRI_Study_open")
    rangpur_volumes = rangpur_base / "semantic_MRs"
    rangpur_labels = rangpur_base / "semantic_labels_only"
    
    if rangpur_base.exists() and rangpur_base.is_dir():
        if rangpur_volumes.exists() and rangpur_labels.exists():
            print(f"✓ Using Rangpur 3D dataset:")
            print(f"  Volumes: {rangpur_volumes}")
            print(f"  Labels: {rangpur_labels}")
            return rangpur_base
        else:
            print(f"WARNING: Rangpur base exists but 3D data not found")
    
    # Fallback to local path
    local_base = Path("./HipMRI_Study_open")
    local_volumes = local_base / "semantic_MRs"
    local_labels = local_base / "semantic_labels_only"
    
    if local_volumes.exists() and local_labels.exists():
        print(f"✓ Using local 3D dataset:")
        print(f"  Volumes: {local_volumes}")
        print(f"  Labels: {local_labels}")
        return local_base
    
    # Try alternative local path (relative to script)
    script_dir = Path(__file__).parent
    alt_local_base = script_dir / "HipMRI_Study_open"
    alt_local_volumes = alt_local_base / "semantic_MRs"
    alt_local_labels = alt_local_base / "semantic_labels_only"
    
    if alt_local_volumes.exists() and alt_local_labels.exists():
        print(f"✓ Using local 3D dataset:")
        print(f"  Volumes: {alt_local_volumes}")
        print(f"  Labels: {alt_local_labels}")
        return alt_local_base
    
    print("ERROR: Could not find 3D dataset paths:")
    print(f"  Rangpur volumes: {rangpur_volumes}")
    print(f"  Rangpur labels: {rangpur_labels}")
    print(f"  Local volumes: {local_volumes}")
    print(f"  Local labels: {local_labels}")
    sys.exit(2)


def validate_dataset(slices_dir):
    """
    Validate dataset structure and integrity.
    
    Args:
        slices_dir: Path to keras_slices_data directory
        
    Returns:
        True if valid, False otherwise
    """
    
    print("\n" + "="*60)
    print("DATASET VALIDATION")
    print("="*60)
    
    slices_dir = Path(slices_dir)
    
    # Check subdirectories
    required_dirs = [
        "keras_slices_train", "keras_slices_seg_train",
        "keras_slices_validate", "keras_slices_seg_validate",
        "keras_slices_test", "keras_slices_seg_test"
    ]
    
    for subdir in required_dirs:
        subdir_path = slices_dir / subdir
        if not subdir_path.exists():
            print(f"ERROR: Missing directory: {subdir_path}")
            return False
        print(f"✓ Found: {subdir}")
    
    # Check NIfTI files
    print("\nScanning NIfTI files...")
    niis = list(slices_dir.rglob("*.nii")) + list(slices_dir.rglob("*.nii.gz"))
    if not niis:
        print("ERROR: No .nii or .nii.gz files found")
        return False
    
    print(f"✓ Found {len(niis)} NIfTI files")
    
    # Test loading first file
    try:
        sample_file = niis[0]
        img = nib.load(str(sample_file))
        print(f"✓ Sample file loads successfully: {img.shape}, dtype: {img.get_data_dtype()}")
    except Exception as e:
        print(f"ERROR: Failed to load sample file: {e}")
        return False
    
    # Get split info
    try:
        train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels = \
            load_keras_slices(slices_dir)
        
        print(f"\nDataset splits:")
        print(f"  Training:   {len(train_imgs)} images")
        print(f"  Validation: {len(val_imgs)} images")
        print(f"  Test:       {len(test_imgs)} images")
        print(f"  Total:      {len(train_imgs) + len(val_imgs) + len(test_imgs)} images")
    except Exception as e:
        print(f"ERROR: Failed to load dataset splits: {e}")
        return False
    
    print("\n✓ Dataset validation PASSED")
    return True


def validate_dataset_3d(base_dir):
    """
    Validate 3D dataset structure and integrity.
    
    Args:
        base_dir: Path to HipMRI_Study_open directory
        
    Returns:
        True if valid, False otherwise
    """
    
    print("\n" + "="*60)
    print("3D DATASET VALIDATION")
    print("="*60)
    
    base_dir = Path(base_dir)
    volumes_dir = base_dir / "semantic_MRs"
    labels_dir = base_dir / "semantic_labels_only"
    
    # Check subdirectories
    if not volumes_dir.exists():
        print(f"ERROR: Missing volumes directory: {volumes_dir}")
        return False
    print(f"✓ Found volumes directory: {volumes_dir}")
    
    if not labels_dir.exists():
        print(f"ERROR: Missing labels directory: {labels_dir}")
        return False
    print(f"✓ Found labels directory: {labels_dir}")
    
    # Check NIfTI files
    print("\nScanning NIfTI files...")
    
    volume_niis = list(volumes_dir.glob("*.nii.gz"))
    label_niis = list(labels_dir.glob("*.nii.gz"))
    
    print(f"✓ Found {len(volume_niis)} volume files")
    print(f"✓ Found {len(label_niis)} label files")
    
    if len(volume_niis) == 0 or len(label_niis) == 0:
        print(f"ERROR: No NIfTI files found")
        return False
    
    # Test loading first files
    try:
        sample_volume = volume_niis[0]
        sample_label = label_niis[0]
        
        vol_img = nib.load(str(sample_volume))
        label_img = nib.load(str(sample_label))
        
        print(f"✓ Sample volume loads: {vol_img.shape}, dtype: {vol_img.get_data_dtype()}")
        print(f"✓ Sample label loads: {label_img.shape}, dtype: {label_img.get_data_dtype()}")
    except Exception as e:
        print(f"ERROR: Failed to load sample files: {e}")
        return False
    
    print("\n✓ 3D Dataset validation PASSED")
    return True


def main():
    """
    Main test routine for 2D/3D Improved UNet training and evaluation.
    
    Supports command-line arguments:
        --mode {2d, 3d}         : Model mode (default: 3d)
        --epochs INT            : Number of epochs (default: 50 for 3d, 50 for 2d)
        --batch_size INT        : Batch size (default: 1)
        --downsample_factor INT : 3D downsampling factor (default: 2)
        --early_stop_patience INT : Early stopping patience (default: 5)
    """
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Train Improved UNet on HipMRI Study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python test.py                              # 3D training, 50 epochs (default)
  python test.py --mode 2d                    # 2D training, 50 epochs
  python test.py --mode 3d --epochs 100       # 3D training, 100 epochs
  python test.py --mode 2d --batch_size 32    # 2D training, batch_size=32
        '''
    )
    
    parser.add_argument('--mode', type=str, default=DEFAULT_MODE, choices=['2d', '3d'],
                        help=f'Model mode (default: {DEFAULT_MODE})')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help=f'Number of training epochs (default: {DEFAULT_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--downsample_factor', type=int, default=DEFAULT_DOWNSAMPLE_FACTOR,
                        help=f'3D downsampling factor (default: {DEFAULT_DOWNSAMPLE_FACTOR})')
    parser.add_argument('--early_stop_patience', type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE,
                        help=f'Early stopping patience (default: {DEFAULT_EARLY_STOPPING_PATIENCE})')
    
    args = parser.parse_args()
    
    # Extract arguments
    mode = args.mode
    num_epochs = args.epochs
    batch_size = args.batch_size
    downsample_factor = args.downsample_factor
    early_stopping_patience = args.early_stop_patience
    
    print("="*60)
    print(f"Improved UNet - HipMRI Study Segmentation ({mode.upper()})")
    print("="*60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    print(f"\nTraining Configuration:")
    print(f"  Mode: {mode.upper()}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Early Stopping Patience: {early_stopping_patience}")
    if mode == '3d':
        print(f"  Downsample Factor: {downsample_factor}")
    
    # Check and set data path
    if mode == '2d':
        data_dir = check_and_set_data_path()
    else:  # 3d mode
        data_dir = check_and_set_data_path_3d()
    
    # Validate dataset
    if mode == '2d':
        if not validate_dataset(data_dir):
            sys.exit(2)
    else:
        if not validate_dataset_3d(data_dir):
            sys.exit(2)
    
    # Set up checkpoint directory (mode-specific)
    checkpoint_dir = Path(f"./checkpoints{'_3d' if mode == '3d' else ''}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    # Prepare training parameters based on mode
    training_params = {
        'data_dir': str(data_dir),
        'mode': mode,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': 1e-3,
        'num_classes': 2,
        'device': device,
        'checkpoint_dir': str(checkpoint_dir),
        'early_stopping_patience': early_stopping_patience
    }
    
    if mode == '3d':
        training_params['downsample_factor'] = downsample_factor
    
    history = train(**training_params)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"Model: {mode.upper()}")
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Best validation Dice: {history['best_val_dice']:.4f}")
    
    if history['test_dice'] is not None:
        print(f"Test Dice scores:")
        for key, value in history['test_dice'].items():
            print(f"  {key}: {value:.4f}")
        
        # Check if prostate label (class 1) meets target
        if history['test_dice'].get('class_1', 0) >= 0.75:
            print(f"\n✓ SUCCESS: Prostate Dice coefficient ({history['test_dice']['class_1']:.4f}) >= 0.75")
        else:
            print(f"\n✗ WARNING: Prostate Dice coefficient ({history['test_dice']['class_1']:.4f}) < 0.75")
            print("  Consider training longer or adjusting hyperparameters")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Training history saved to: {checkpoint_dir / 'training_history.json'}")


if __name__ == '__main__':
    main()

