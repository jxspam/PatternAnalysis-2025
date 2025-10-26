# 2D Improved UNet for HipMRI Study Prostate Segmentation

## Problem Statement and Algorithm

**Problem**: Accurate segmentation of prostate structures from 2D MRI slices for clinical diagnosis and treatment planning.

**Solution**: Improved UNet architecture - an encoder-decoder CNN with:

- **Skip connections**: Concatenate encoder outputs with decoder upsampled features for better feature preservation
- **Batch normalization**: Stabilizes training and enables higher learning rates
- **Dropout regularization (0.5)**: Prevents overfitting by randomly disabling 50% of activations
- **Dice Loss**: Directly optimizes segmentation metric, handles class imbalance better than cross-entropy

**Architecture**:

```
Input (1, H, W) → Encoder (4 levels, 64→512 filters, MaxPool)
                → Bottleneck → Decoder (4 levels, UpConv + Skip connections)
                → Output (2, H, W) → Softmax → [Background, Prostate]
```

## Dataset and Preprocessing

**HipMRI Study 2D Slices**:

- 25,320 NIfTI files, 256×128 pixels, 2 classes (background: 0, prostate: 1)
- Pre-split: Train 60% | Validation 20% | Test 20%

**Preprocessing**:

1. **Normalization**: Min-max to [0, 1] for consistent intensity ranges
2. **Augmentation** (training only):
   - Random rotation: ±15°
   - Random horizontal/vertical flip: 50% probability
3. **Consistent augmentation**: Same seed for image and label transforms

**Justification**:

- Normalization improves convergence across different MRI scans
- Augmentation increases dataset diversity and prevents overfitting
- Train/val/test split (60/20/20) provides robust evaluation

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Training with Test Script (Recommended)

```bash
# Automatically detects Rangpur GPU cluster or uses local data
python test.py
```

### Training

```bash
python train.py --data_dir ./HipMRI_Study_open/keras_slices_data --epochs 1 --batch_size 1
```

### Inference

```python
import torch
from modules import ImprovedUNet2D
from predict import load_model, predict_single_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("checkpoints/best_model_epoch_1.pt", device=device)
prediction, image = predict_single_image(model, "path/to/image.nii.gz", device=device)
# prediction shape: (H, W) with class indices [0, 1]
```

## Implementation Files

| File            | Purpose                                                                              |
| --------------- | ------------------------------------------------------------------------------------ |
| `modules.py`    | ConvBlock, UpConvBlock, ImprovedUNet2D, DiceLoss, dice_coefficient                   |
| `dataset.py`    | MedicalImageDataset, load_keras_slices(), create_data_loaders()                      |
| `train.py`      | train_epoch(), validate_epoch(), test_epoch(), train()                               |
| `predict.py`    | load_model(), predict_single_image(), evaluate_predictions(), visualize_prediction() |
| `load_nifti.py` | Example: Loading NIfTI files using nibabel and nilearn                               |
| `test.py`       | Automated test suite for Rangpur GPU cluster                                         |

## Results

**Training Configuration**: Epochs=1, Batch Size=1, Learning Rate=1e-3, Adam Optimizer

| Metric                   | Value        |
| ------------------------ | ------------ |
| Training Loss            | 0.0790       |
| Validation Dice          | 0.9112       |
| Test Dice (Background)   | 0.9593       |
| **Test Dice (Prostate)** | **0.8893 ✓** |
| Test Dice (Average)      | 0.9243       |

**Status**: ✓ Exceeds 0.75 target. Prostate segmentation Dice: **88.93%**

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

## References

1. Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
2. Dice, L. R. (1945). Measures of Ecologic Association Between Species. Ecology.
3. Milletari et al. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. 3DV.

---

**Author**: Jun Xi Koh (48225421) | **Course**: COMP3710 | **Date**: 2025-10-27 | **Version**: 3.0
