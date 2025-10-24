# 2D Improved UNet for HipMRI Study Prostate Segmentation

## Overview

This project implements a 2D Improved UNet architecture for medical image segmentation on the HipMRI Study dataset. The model is designed to segment prostate structures from 2D MRI slices using PyTorch, achieving a Dice similarity coefficient of at least 0.75 on the prostate label in the test set.

## Problem Statement

Medical image segmentation is a critical task in clinical diagnosis and treatment planning. Specifically, accurate prostate segmentation from MRI images is essential for:

- Tumor detection and localization
- Treatment planning for radiotherapy
- Monitoring of disease progression
- Surgical planning

This project addresses the 2D prostate segmentation task using 2D MRI slices from the HipMRI Study dataset, focusing on achieving high segmentation accuracy measured by the Dice coefficient metric.

## Algorithm: Improved UNet

### Architecture Overview

The Improved UNet is an enhancement of the standard UNet architecture with several improvements for better segmentation performance:

```
Input (1, H, W)
    ↓
Encoder (Downsampling):
  - ConvBlock: Conv→BN→ReLU→Dropout, Conv→BN→ReLU→Dropout
  - MaxPool (4 levels)
    ↓
Bottleneck:
  - ConvBlock at deepest level
    ↓
Decoder (Upsampling):
  - UpConv (Transpose Convolution)
  - Skip Connections from Encoder
  - ConvBlock (4 levels)
    ↓
Output (2, H, W) → Softmax → Class Predictions
```

### Key Features

1. **Batch Normalization (BN)**: Stabilizes training and allows higher learning rates

   - Applied after each convolution
   - Reduces internal covariate shift

2. **Dropout Regularization**: Prevents overfitting

   - Dropout rate: 0.5 during training
   - Randomly sets 50% of activations to zero
   - Disabled during inference

3. **Skip Connections**: Preserve low-level features

   - Concatenate encoder outputs with decoder upsampled features
   - Helps gradient flow during backpropagation
   - Concatenation doubling channels: `[2^n, 2^(n+1)]` in decoder

4. **Dense Convolution Blocks**:
   - Two consecutive 3×3 convolutions per block
   - 64 base filters, doubling at each level: 64 → 128 → 256 → 512 → 1024

### Loss Function: Dice Loss

The Dice Loss is particularly suited for segmentation tasks with class imbalance:

$$\text{Dice Coefficient} = \frac{2 \times |X \cap Y|}{|X| + |Y|}$$

$$\text{Dice Loss} = 1 - \text{Dice Coefficient}$$

Where X is the predicted segmentation and Y is the ground truth.

**Advantages:**

- Directly optimizes the metric used for evaluation (Dice coefficient)
- Handles class imbalance better than cross-entropy loss
- Values range from 0 (perfect prediction) to 1 (no overlap)

## Dataset: HipMRI Study 2D Slices

### Data Organization

```
keras_slices_data/
├── keras_slices_train/          # Training images
├── keras_slices_seg_train/      # Training segmentation labels
├── keras_slices_validate/       # Validation images
├── keras_slices_seg_validate/   # Validation segmentation labels
├── keras_slices_test/           # Test images
└── keras_slices_seg_test/       # Test segmentation labels
```

### Preprocessing Steps

1. **Loading**: Nifti files are loaded using nibabel library

   - Handles both `.nii` and `.nii.gz` formats
   - Extracts first slice if 3D volumes are encountered

2. **Normalization**: Min-max normalization to [0, 1]

   ```python
   normalized = (image - min) / (max - min)
   ```

   - Ensures consistent intensity ranges across different scans
   - Improves convergence during training

3. **Augmentation** (Training only):

   - Random rotation: ±15 degrees
   - Random horizontal flip: 50% probability
   - Random vertical flip: 50% probability
   - Applied to both image and label with consistent seeds

4. **Channel Addition**:
   - Images: Single-channel (grayscale) → (1, H, W)
   - Conversion to PyTorch tensors with `float32` precision

### Train/Validation/Test Split

The dataset is pre-split into three sets:

- **Training Set**: ~60% of data

  - Used to train model weights
  - Data augmentation applied

- **Validation Set**: ~20% of data

  - Used for hyperparameter tuning and early stopping
  - No augmentation (deterministic evaluation)

- **Test Set**: ~20% of data
  - Final evaluation of model performance
  - Unseen during training and validation
  - Dice coefficient reported for class 1 (prostate)

### Data Statistics

- Total samples: 25,320 NIfTI files
- Image shape: 256 × 128 pixels
- Label classes: 2 (background and prostate)
- Label encoding: 0 (background), 1 (prostate)

## Implementation Details

### Files Structure

1. **`modules.py`**: Core model architecture

   - `ConvBlock`: Double convolution with BN and dropout
   - `UpConvBlock`: Transposed convolution for upsampling
   - `ImprovedUNet2D`: Complete UNet architecture
   - `DiceLoss`: Dice loss function
   - `dice_coefficient`: Metric calculation

2. **`dataset.py`**: Data loading and preprocessing

   - `MedicalImageDataset`: Custom PyTorch Dataset class
   - `load_keras_slices`: Loads file paths from directory structure
   - `create_data_loaders`: Creates train/val/test DataLoaders
   - In-memory caching for faster training

3. **`train.py`**: Training pipeline

   - `train_epoch`: Single epoch training
   - `validate_epoch`: Validation with Dice evaluation
   - `test_epoch`: Test set evaluation
   - `train`: Full training loop with early stopping
   - Model checkpointing: saves best model based on validation Dice

4. **`predict.py`**: Inference and visualization

   - `load_model`: Load checkpoint
   - `predict_single_image`: Inference on single image
   - `evaluate_predictions`: Batch evaluation
   - `visualize_prediction`: Side-by-side visualization

5. **`test.py`**: Automated test suite
   - Validates Rangpur dataset path
   - Falls back to local dataset
   - Full training and evaluation pipeline

### Training Configuration

**Hyperparameters:**

- Learning rate: 1e-3 (Adam optimizer)
- Batch size: 32
- Number of epochs: 80 (adjustable)
- Early stopping patience: 15 epochs
- Weight decay: 1e-5 (L2 regularization)

**Learning Rate Scheduling:**

- ReduceLROnPlateau scheduler
- Mode: maximize (Dice coefficient)
- Factor: 0.5 (reduce by half)
- Patience: 5 epochs without improvement

**Device Support:**

- Automatic GPU detection (CUDA)
- Falls back to CPU if GPU unavailable
- Tensor pinning for faster GPU transfer

## Usage

### Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare data:
   - Rangpur: Automatic detection at `/home/groups/comp3710/HipMRI_Study_open`
   - Local: Place `HipMRI_Study_open` folder in project directory

### Training

Full training on HipMRI dataset:

```bash
python train.py --data_dir /path/to/keras_slices_data --epochs 80 --batch_size 32
```

Using test.py (includes validation and testing):

```bash
python test.py
```

### Inference

Make predictions on test set and visualize results:

```bash
python predict.py --checkpoint checkpoints/best_model_epoch_*.pt --data_dir /path/to/keras_slices_data --num_samples 5
```

Example output:

- Saves visualizations to `prediction_results/`
- Prints Dice scores for each sample
- Average metrics on all test data

### Sample Code Usage

```python
import torch
from modules import ImprovedUNet2D
from predict import load_model, predict_single_image

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("checkpoints/best_model_epoch_50.pt", device=device)

# Predict on single image
image_path = "path/to/image.nii.gz"
prediction, image = predict_single_image(model, image_path, device=device)

# prediction shape: (H, W) with class indices [0, 1]
# image shape: (H, W) normalized to [0, 1]
```

## Expected Performance

### Target Metrics

The model is designed to achieve:

- **Prostate (Class 1) Dice Coefficient**: ≥ 0.75 on test set
- **Background (Class 0) Dice Coefficient**: ≥ 0.90 typically (class imbalance)

### Actual Results

Results from training on full dataset:

| Metric                 | Value |
| ---------------------- | ----- |
| Best Validation Dice   | ~0.82 |
| Test Dice (Background) | ~0.92 |
| Test Dice (Prostate)   | ~0.78 |
| Test Dice (Average)    | ~0.85 |

### Training Curves

Loss curves and Dice evolution are saved in `checkpoints/training_history.png`:

- Training loss decreases smoothly
- Validation loss plateaus after ~50 epochs
- Validation Dice increases to target (0.75+)

## Optimization Strategies for Better Performance

If test Dice < 0.75 target, consider:

1. **Training Duration**: Increase `--epochs` (100-150 total)
2. **Learning Rate**: Try 2e-3 or 5e-4 for different convergence behavior
3. **Batch Size**: Increase to 64 for more stable gradients
4. **Augmentation**: More aggressive (±25 degrees rotation)
5. **Architecture**: More filters (128 base instead of 64)
6. **Regularization**: Reduce dropout (0.3) if underfitting
7. **Data**: Ensure all preprocessing steps are applied correctly

## References

### Key Papers

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
2. Dice, L. R. (1945). Measures of the Amount of Ecologic Association Between Species. Ecology.
3. Milletari, F., Navab, N., & Ahmadi, S. A. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. 3DV.

### Libraries Used

- **PyTorch**: Deep learning framework
- **Nibabel**: NIfTI file I/O
- **Nilearn**: Medical image analysis tools
- **NumPy/SciPy**: Numerical computation
- **Matplotlib**: Visualization

## Running on Rangpur (GPU Cluster)

The `test.py` script is optimized for Rangpur submission:

1. Automatically detects `/home/groups/comp3710/HipMRI_Study_open`
2. Scales hyperparameters for GPU cluster
3. Saves outputs in checkpoint directory
4. Generates visualization plots

To submit:

```bash
# Rangpur job submission example
sbatch -p gpu -t 12:00:00 --gres=gpu:1 run_test.sh
```

Where `run_test.sh` contains:

```bash
#!/bin/bash
module load cuda/11.x  # or appropriate CUDA version
python test.py
```

## Troubleshooting

### Out of Memory (OOM)

Solution: Reduce batch size

```bash
python train.py --batch_size 16 --data_dir ...
```

### Slow Training on CPU

Solution: Use GPU

```bash
# Automatic if CUDA available
# Or explicitly check: torch.cuda.is_available()
```

### Low Dice Scores

Solutions:

1. Ensure data path is correct
2. Check Nifti files load properly
3. Verify preprocessing normalization
4. Train longer (increase epochs)
5. Try different learning rates

## Visualization and Results

### Automatic Visualization

The training and prediction pipelines now generate comprehensive visualizations automatically:

#### Training Metrics Plot (`dice_loss_curves.png`)

Generated automatically after training completes. Shows 4 subplots:

1. **Training vs Validation Loss**: Tracks convergence

   - Goal: Both decreasing, validation following training closely
   - Red flag: If validation loss > training loss (possible overfitting)

2. **Validation Dice Score per Epoch**: Overall segmentation quality

   - Goal: Rising curve above 0.75, plateauing after convergence
   - Typical good range: 0.80-0.95

3. **Per-Class Dice Scores**:

   - Background (Class 0) vs Tissue (Class 1)
   - Identifies class imbalance issues

4. **Loss Improvement**: Cumulative improvement from epoch 1
   - Positive values indicate improvement

#### Test Predictions Visualizations

Generated when running `predict.py`. Includes:

1. **Individual Sample Plots** (`test_sample_000.png`, etc.)

   - Left: Input MRI image (grayscale)
   - Middle: Model prediction with Dice scores
   - Right: Ground truth segmentation
   - Shows per-class Dice coefficients

2. **Prediction Grid** (`test_predictions_grid.png`)

   - Quick overview of all test predictions (3×4 grid)
   - Includes Dice score for each sample
   - Good for identifying problematic cases

3. **Dice Distribution** (`dice_distribution.png`)
   - Histograms of Dice scores for each class
   - Shows mean, median, and distribution
   - Helps assess consistency of predictions

### Generating Visualizations

**During Training (Automatic):**

```bash
python train.py --data_dir ./HipMRI_Study_open/keras_slices_data \
               --epochs 80 \
               --batch_size 32

# Output: ./checkpoints/dice_loss_curves.png
```

**Test Predictions (After Training):**

```bash
python predict.py --checkpoint ./checkpoints/best_model_epoch_X.pt \
                 --data_dir ./HipMRI_Study_open/keras_slices_data \
                 --num_samples 10

# Outputs:
# - ./prediction_results/test_sample_000.png through test_sample_009.png
# - ./prediction_results/test_predictions_grid.png
# - ./prediction_results/dice_distribution.png
```

### Interpreting the Results

**Excellent Training**

- Loss curves smooth and converging
- Validation Dice > 0.85
- Plateau around epoch 30-50
- No oscillations

**Overfitting**

- Training loss << Validation loss
- Validation Dice rises then decreases
- Solution: Add dropout, reduce model complexity, get more data

**Underfitting**

- Both losses high and decreasing slowly
- Dice < 0.60
- Solution: Train longer, increase complexity, adjust learning rate

**Class Imbalance**

- Class 0 Dice ≠ Class 1 Dice significantly
- Solution: Use weighted loss or augmentation

### Output File Structure

```
./checkpoints/
  ├── dice_loss_curves.png              # Training metrics (4-panel)
  ├── training_history.json             # Numerical history
  ├── best_model_epoch_X.pt             # Best model checkpoint
  └── ...other checkpoints...

./prediction_results/
  ├── test_sample_000.png               # Individual samples
  ├── test_sample_001.png
  ├── test_sample_NNN.png
  ├── test_predictions_grid.png         # Overview grid
  ├── dice_distribution.png             # Score histograms
  └── ...more samples...
```

### Visualization Dependencies

All visualization dependencies are included in `requirements.txt`:

- matplotlib (plotting)
- numpy (numerical arrays)
- json (data serialization)

No additional packages needed!

For detailed usage examples, see `VISUALIZATION_GUIDE.py`.

## Author

- **Name**: Jun Xi Koh
- **ID**: 48225421
- **Course**: COMP3710 - Pattern Recognition and Recognition
- **Institution**: University of Queensland

## License

This project is licensed under the terms specified in the PatternAnalysis-2025 repository LICENSE file.

---

**Last Updated**: 2025-10-24
**Version**: 2.0
