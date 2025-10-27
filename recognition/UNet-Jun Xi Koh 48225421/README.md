# Improved UNet for HipMRI Study - 2D and 3D Segmentation

## Overview

This project implements both **2D and 3D Improved UNet architectures** for prostate segmentation from MRI data:

- **2D UNet**: Segment 2D slices with 88.93% Dice coefficient
- **3D UNet**: Segment 3D volumetric data with downsampling for memory efficiency

Both models achieve excellent segmentation performance on the HipMRI Study dataset.

## Problem Statement and Algorithm

**Problem**: Accurate segmentation of prostate structures from MRI for clinical diagnosis and treatment planning.

**Solution**: Improved UNet architecture - an encoder-decoder CNN with:

- **Skip connections**: Concatenate encoder outputs with decoder upsampled features for better feature preservation
- **Batch normalization**: Stabilizes training and enables higher learning rates
- **Dropout regularization (0.5)**: Prevents overfitting by randomly disabling 50% of activations
- **Weighted Dice Loss**: Directly optimizes segmentation metric with **class weighting** to handle severe class imbalance
- **Class Imbalance Handling** (3D): Background voxels ~7x more frequent than prostate voxels
  - Solution: Apply inverse frequency weights [0.25, 1.75] to penalize mistakes on minority class
  - Result: Prostate Dice improved from 24% → 93%

**2D Architecture**:

```
Input (1, H, W) → Encoder (4 levels, 64→512 filters) → Bottleneck
                → Decoder (4 levels, UpConv + Skip) → Output (2, H, W)
```

**3D Architecture**:

```
Input (1, D, H, W) → Encoder (3 levels, 32→256 filters) → Bottleneck
                   → Decoder (3 levels, UpConv + Skip) → Output (2, D, H, W)
```

(Smaller filters and fewer levels for 3D to fit in memory with downsampling)

## Datasets

### 2D Dataset: HipMRI Study 2D Slices

- **25,320 NIfTI files**, 256×128 pixels, 2 classes (background: 0, prostate: 1)
- Pre-split: Train 60% | Validation 20% | Test 20%
- Path: `/home/groups/comp3710/HipMRI_Study_open/keras_slices_data`

### 3D Dataset: HipMRI Study Volumetric Data

- **38 patients & 211 3D MRI volumes**, variable dimensions, 2 classes
- Downsampled 2x in each dimension for memory efficiency
- Split: Train 60% | Validation 20% | Test 20%
- Paths:
  - Volumes: `/home/groups/comp3710/HipMRI_Study_open/semantic_MRs`
  - Labels: `/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only`

### Preprocessing

1. **Normalization**: Min-max to [0, 1] for consistent intensity ranges
2. **Augmentation** (2D training only):
   - Random rotation: ±15°
   - Random horizontal/vertical flip: 50% probability
3. **3D Downsampling**: 2x reduction per dimension using trilinear interpolation
4. **Consistent augmentation**: Same seed for image and label transforms

**Justification**:

- Normalization improves convergence across different MRI scans
- Augmentation increases dataset diversity and prevents overfitting
- 3D downsampling reduces memory requirements while preserving information
- Train/val/test split (60/20/20) provides robust evaluation

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### 2D UNet Training (2D Slices)

```bash
# Test script (recommended) - automatically detects Rangpur or local data
python test2d.py

# Or direct training
python train.py --data_dir ./HipMRI_Study_open/keras_slices_data --epochs 1 --batch_size 1
```

### 3D UNet Training (Volumetric Data)

```bash
# Test script for 3D - automatically detects Rangpur or local data
python test3d.py

# Or direct training
python train3d.py --data_dir ./HipMRI_Study_open --epochs 1 --batch_size 1 --downsample_factor 2
```

### 2D Inference

```python
import torch
from modules import ImprovedUNet2D
from predict import load_model, predict_single_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("checkpoints/best_model_epoch_1.pt", device=device)
prediction, image = predict_single_image(model, "path/to/slice.nii.gz", device=device)
# prediction shape: (H, W) with class indices [0, 1]
```

### 3D Inference

```python
import torch
from modules import ImprovedUNet3D
from predict3d import load_model_3d, predict_single_volume

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model_3d("checkpoints_3d/best_model_epoch_1.pt", device=device)
prediction, downsampled_vol, original_vol = predict_single_volume(
    model, "path/to/volume.nii.gz", downsample_factor=2, device=device
)
# prediction shape: (D, H, W) with class indices [0, 1]
```

## Implementation Files

**Unified Architecture** (consolidated 2D and 3D implementations):

| File            | Purpose                                                                                                     |
| --------------- | ----------------------------------------------------------------------------------------------------------- |
| `modules.py`    | 2D/3D components: ConvBlock2D/3D, UpConvBlock2D/3D, ImprovedUNet2D/3D, shared DiceLoss with class weighting |
| `dataset.py`    | **Unified** 2D and 3D data loading via `mode='2d'/'3d'` parameter:                                          |
|                 | - 2D: MedicalImageDataset, load_keras_slices(), for 2D slices                                               |
|                 | - 3D: VolumetricDataset, load_semantic_data(), with automatic downsampling (2x)                             |
|                 | - Unified: create_data_loaders(mode='2d'/'3d') wrapper function                                             |
| `train.py`      | **Unified** training script supporting both 2D and 3D via `mode='2d'/'3d'` parameter:                       |
|                 | - train(mode='2d'/'3d', ...): Automatically selects model, loss, and data loader                            |
|                 | - Supports argparse: `--mode 2d/3d --epochs 50 --downsample_factor 2`                                       |
|                 | - Class weights [0.25, 1.75] applied only for 3D to handle imbalance                                        |
| `predict.py`    | **Unified** inference script supporting both 2D and 3D via `mode='2d'/'3d'` parameter:                      |
|                 | - load_model(checkpoint_path, mode='2d'/'3d'): Load appropriate model                                       |
|                 | - predict_single_image(): 2D inference                                                                      |
|                 | - predict_single_volume(): 3D inference with downsampling                                                   |
|                 | - evaluate_predictions(): Compute Dice scores (auto-selects 2D or 3D function)                              |
| `load_nifti.py` | Example: Loading NIfTI files using nibabel and nilearn                                                      |
| `test.py`       | Automated test suite for 2D Rangpur/local execution (mode='2d')                                             |
| `test2d.py`     | Alternative test suite for 2D execution                                                                     |
| `test3d.py`     | Automated test suite for 3D Rangpur/local execution (mode='3d')                                             |

**Usage Examples**:

```bash
# 2D Training
python train.py --mode 2d --data_dir ./HipMRI_Study_open/keras_slices_data --epochs 50

# 3D Training
python train.py --mode 3d --data_dir ./HipMRI_Study_open --epochs 50 --downsample_factor 2

# 2D Prediction
python predict.py --mode 2d --checkpoint ./checkpoints/best_model.pt --image_path input.nii.gz

# 3D Prediction
python predict.py --mode 3d --checkpoint ./checkpoints_3d/best_model.pt --volume_path volume.nii.gz
```

## Results

### 2D UNet Results

**Training Configuration**: Epochs=1, Batch Size=1, Learning Rate=1e-3, Adam Optimizer

| Metric                   | Value        |
| ------------------------ | ------------ |
| Training Loss            | 0.0790       |
| Validation Dice          | 0.9112       |
| Test Dice (Background)   | 0.9593       |
| **Test Dice (Prostate)** | **0.8893 ✓** |
| Test Dice (Average)      | 0.9243       |

**Status**: ✓ Exceeds 0.75 target. Prostate segmentation Dice: **88.93%**

### 3D UNet Results

**Training Configuration**: Epochs=50 (Early Stopped at Epoch 2), Batch Size=1, Learning Rate=1e-3, Downsample Factor=2x, Early Stopping Patience=2, Adam Optimizer

**Class Weights for Loss**: [0.25, 1.75] - Applied to handle class imbalance (prostate is ~7x less frequent)

| Metric                     | Value      | Status          |
| -------------------------- | ---------- | --------------- |
| Training Loss (Best)       | 0.0896     | ✓ Converged     |
| Validation Loss (Best)     | 0.0596     | ✓ Stable        |
| Best Epoch                 | 2          | ✓ Early Stop     |
| Validation Dice (Avg)      | 0.9651     | ✓ Excellent     |
| **Test Dice (Background)** | **0.9948** | ✓ Exceeds 0.7   |
| **Test Dice (Prostate)**   | **0.9369** | ✓ Exceeds 0.7   |
| **Test Dice (Average)**    | **0.9659** | ✓ **EXCELLENT** |

**Status**: ✓✓✓ All metrics far exceed ≥0.7 target. Prostate segmentation Dice: **93.69%** | Background: **99.48%**

**Key Improvements Over Initial Attempt**:

- Initial attempt (no weights): Class 1 Dice = 0.24 (24%)
- With weighted loss: Class 1 Dice = 0.93 (93%) **✓ 388% improvement**
- Class weighting [0.25, 1.75] penalizes underrepresented class more
- Training with 50 epochs allows proper convergence

**3D-specific Implementation Details**:

- **Input**: 3D volumetric MRI (D, H, W) instead of 2D slices
- **Downsampling**: 2x reduction per dimension → ~8x memory reduction (2³)
- **Architecture**: 3 encoder/decoder levels (vs 4 for 2D) to fit in GPU memory
- **Filters**: 32 base filters (vs 64 for 2D) for memory efficiency
- **Interpolation**: Trilinear for upsampling and spatial matching
- **Data handling**: VolumetricDataset with caching for efficient loading
- **Loss function**: Weighted Dice Loss with class weights [0.25, 1.75] to handle severe class imbalance

### Example Output

**Input**: MRI slice (256×128, single-channel grayscale)
**Output**: Segmentation mask (256×128, two classes)

Visualizations saved to `prediction_results/`:

- `test_sample_000.png` - Side-by-side input/prediction/ground-truth
- `test_predictions_grid.png` - Overview grid of all predictions
- `dice_distribution.png` - Histogram of Dice scores per class

Training metrics saved to `checkpoints/`:

- `dice_loss_curves.png` - 4-panel plot: loss, Dice per epoch, per-class Dice
- `training_history.json` - Numerical results

## Resource Optimization

**Storage Constraint Solution** (10GB limit):

- Removed unnecessary packages: TensorFlow, Keras, scipy, scikit-image, opencv-python, pandas, nilearn
- Final dependencies: torch, torchvision, nibabel, numpy, matplotlib, tqdm (~2-3GB)
- Training: 1 epoch, batch size 1 → minimal memory, excellent quality
- **3D optimization**: Downsampling (2x) reduces memory footprint by ~8x per volume

## References

1. Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
2. Dice, L. R. (1945). Measures of Ecologic Association Between Species. Ecology.
3. Milletari et al. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. 3DV.

---

**Author**: Jun Xi Koh (48225421) | **Course**: COMP3710 | **Date**: 2025-10-27 | **Version**: 4.0 (2D + 3D UNet)
