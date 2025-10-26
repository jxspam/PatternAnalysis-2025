"""
test3d.py - Test script for 3D Improved UNet on HipMRI Study

This script:
1. Validates dataset paths (Rangpur or local)
2. Trains the 3D Improved UNet model
3. Evaluates on test set with Dice coefficient for all labels
4. Saves results and visualizations

For Rangpur (GPU): Uses /home/groups/comp3710/HipMRI_Study_open
For Local: Uses ./HipMRI_Study_open if Rangpur path not available
"""

from pathlib import Path
import sys
import torch
import nibabel as nib

from train import train
from dataset import load_semantic_data
from visualization import generate_all_3d_visualizations

# Fixed configuration parameters for 3D (memory-efficient)
EPOCHS = 50
BATCH_SIZE = 1
EARLY_STOPPING_PATIENCE = 2
NUM_WORKERS = 0
DOWNSAMPLE_FACTOR = 2  # Downsample 3D volumes by 2x to fit in memory

def check_and_set_data_path():
    """
    Check for dataset availability on Rangpur or fallback to local.
    
    Returns:
        Path to HipMRI_Study_open directory
    """
    
    # Try Rangpur path first
    rangpur_base = Path("/home/groups/comp3710/HipMRI_Study_open")
    
    if rangpur_base.exists() and rangpur_base.is_dir():
        print(f"✓ Using Rangpur dataset: {rangpur_base}")
        return rangpur_base
    
    # Fallback to local path
    local_base = Path("./HipMRI_Study_open")
    if local_base.exists() and local_base.is_dir():
        print(f"✓ Using local dataset: {local_base}")
        return local_base
    
    # Try alternative local path (relative to script)
    script_dir = Path(__file__).parent
    alt_local_base = script_dir / "HipMRI_Study_open"
    if alt_local_base.exists() and alt_local_base.is_dir():
        print(f"✓ Using local dataset: {alt_local_base}")
        return alt_local_base
    
    print("ERROR: Could not find dataset paths:")
    print(f"  Rangpur: {rangpur_base}")
    print(f"  Local: {local_base}")
    print(f"  Alt Local: {alt_local_base}")
    sys.exit(2)


def validate_3d_dataset(base_dir):
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
    
    # Check required subdirectories
    volume_dir = base_dir / "semantic_MRs"
    label_dir = base_dir / "semantic_labels_only"
    
    if not volume_dir.exists():
        print(f"ERROR: Missing volume directory: {volume_dir}")
        return False
    print(f"✓ Found volume directory: {volume_dir}")
    
    if not label_dir.exists():
        print(f"ERROR: Missing label directory: {label_dir}")
        return False
    print(f"✓ Found label directory: {label_dir}")
    
    # Check NIfTI files
    print("\nScanning NIfTI files...")
    volumes = list(volume_dir.glob("*.nii")) + list(volume_dir.glob("*.nii.gz"))
    labels = list(label_dir.glob("*.nii")) + list(label_dir.glob("*.nii.gz"))
    
    if not volumes:
        print(f"ERROR: No .nii or .nii.gz files found in {volume_dir}")
        return False
    print(f"✓ Found {len(volumes)} volume files")
    
    if not labels:
        print(f"ERROR: No .nii or .nii.gz files found in {label_dir}")
        return False
    print(f"✓ Found {len(labels)} label files")
    
    # Test loading first file
    try:
        sample_volume = volumes[0]
        img = nib.load(str(sample_volume))
        shape = img.shape
        dtype = img.get_data_dtype()
        print(f"✓ Sample volume loads successfully: shape={shape}, dtype={dtype}")
    except Exception as e:
        print(f"ERROR: Failed to load sample volume: {e}")
        return False
    
    try:
        sample_label = labels[0]
        lbl_img = nib.load(str(sample_label))
        lbl_shape = lbl_img.shape
        print(f"✓ Sample label loads successfully: shape={lbl_shape}")
    except Exception as e:
        print(f"ERROR: Failed to load sample label: {e}")
        return False
    
    print("\n✓ 3D Dataset validation PASSED")
    return True


def main():
    """
    Main test routine for 3D Improved UNet training and evaluation.
    """
    
    print("="*60)
    print("3D Improved UNet - HipMRI Study Volumetric Segmentation")
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
    print(f"  Downsample Factor: {DOWNSAMPLE_FACTOR}x")
    
    # Check and set data path
    data_dir = check_and_set_data_path()
    
    # Validate dataset
    if not validate_3d_dataset(data_dir):
        sys.exit(2)
    
    # Set up checkpoint directory
    checkpoint_dir = Path("./checkpoints_3d")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING 3D UNET")
    print("="*60)
    
    history = train(
        data_dir=str(data_dir),
        mode='3d',
        num_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=1e-3,
        num_classes=2,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        downsample_factor=DOWNSAMPLE_FACTOR
    )
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Best validation Dice (avg): {history['best_val_dice']:.4f}")
    print(f"Downsample factor used: {history['downsample_factor']}x")
    
    if history['test_dice'] is not None:
        print(f"\nTest Dice scores:")
        for key, value in history['test_dice'].items():
            print(f"  {key}: {value:.4f}")
        
        # Check if all labels meet minimum Dice of 0.7
        test_dice = history['test_dice']
        all_above_threshold = all(
            test_dice.get(f'class_{c}', 0) >= 0.7 
            for c in [0, 1]
        )
        
        if all_above_threshold:
            print(f"\n✓ SUCCESS: All labels exceed Dice coefficient >= 0.7")
            print(f"  Class 0 (Background): {test_dice.get('class_0', 0):.4f}")
            print(f"  Class 1 (Prostate): {test_dice.get('class_1', 0):.4f}")
        else:
            print(f"\n✗ WARNING: Some labels below Dice coefficient 0.7")
            print(f"  Class 0 (Background): {test_dice.get('class_0', 0):.4f}")
            print(f"  Class 1 (Prostate): {test_dice.get('class_1', 0):.4f}")
            print("  Consider training longer or adjusting hyperparameters")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Training history saved to: {checkpoint_dir / 'training_history_3d.json'}")
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    try:
        generate_all_3d_visualizations(
            checkpoint_dir=str(checkpoint_dir),
            data_dir=str(data_dir),
            device=device,
            num_detailed_samples=10,
            downsample_factor=DOWNSAMPLE_FACTOR
        )
        print("\n✓ Visualizations generated successfully!")
    except Exception as e:
        print(f"\n✗ Error generating visualizations: {e}")
        print("  (This is non-fatal, but visualizations will not be available)")


if __name__ == '__main__':
    main()
