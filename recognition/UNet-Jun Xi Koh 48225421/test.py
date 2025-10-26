"""
test.py - Test script for 2D Improved UNet on HipMRI Study

This script:
1. Validates dataset paths (Rangpur or local)
2. Trains the 2D Improved UNet model
3. Evaluates on test set with Dice coefficient
4. Saves results and visualizations

For Rangpur (GPU): Uses /home/groups/comp3710/HipMRI_Study_open
For Local: Uses ./HipMRI_Study_open if Rangpur path not available
"""

from pathlib import Path
import sys
import torch
import nibabel as nib

from train import train
from predict import load_model, evaluate_predictions, visualize_prediction
from dataset import load_keras_slices

# Fixed configuration parameters for minimal resource usage
EPOCHS = 1
BATCH_SIZE = 1
EARLY_STOPPING_PATIENCE = 2
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


def main():
    """
    Main test routine for 2D Improved UNet training and evaluation.
    """
    
    print("="*60)
    print("2D Improved UNet - HipMRI Study Segmentation")
    print("="*60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    
    # Check and set data path
    data_dir = check_and_set_data_path()
    
    # Validate dataset
    if not validate_dataset(data_dir):
        sys.exit(2)
    
    # Set up checkpoint directory
    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    history = train(
        data_dir=str(data_dir),
        num_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=1e-3,
        num_classes=2,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
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

