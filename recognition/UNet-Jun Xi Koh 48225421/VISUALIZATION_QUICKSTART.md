# Quick Reference: 3D UNet Visualization

## One-Command Training + Visualization

```bash
python test3d.py
```

This automatically:

1. Trains the 3D UNet model
2. Generates training curves (Loss and Dice)
3. Creates 10-sample-per-PNG test comparisons
4. Produces detailed multi-slice analysis

**All output goes to**: `./results/`

---

## Generated Outputs

### `training_curves_3d.png`

Shows model learning across epochs:

- **Left graph**: How train/val loss decreased
- **Right graph**: Dice coefficient improvement (target: > 0.7)

### `test_predictions_batch_*.png`

Quick visual inspection of predictions:

- **Red regions**: Model's predicted prostate
- **Blue outlines**: Ground truth prostate
- **Gray background**: Input MRI slice

### `predictions_detailed/sample_*.png`

Deep analysis of individual samples:

- 5 slices per sample
- Shows input, predictions, ground truth, and overlays
- Helps identify where model performs well/poorly

---

## Understanding Quality

| What You Want to See           | What It Means                    |
| ------------------------------ | -------------------------------- |
| Red fills blue outline exactly | Perfect prediction ✓             |
| Red extends beyond blue        | Over-predicted (false positive)  |
| Red doesn't fill entire blue   | Under-predicted (false negative) |
| Dice score > 0.90              | Excellent segmentation           |
| Consistent across all slices   | Model generalizes well           |

---

## Common Tasks

### Just generate visualizations (keep existing model)

```bash
python visualization.py --data_dir ./HipMRI_Study_open
```

### Generate with more detailed samples (20 instead of 10)

```bash
python visualization.py --data_dir ./HipMRI_Study_open --num_detailed 20
```

### Use CPU instead of GPU

```bash
python test3d.py  # Automatic GPU detection
# OR
python visualization.py --data_dir ./HipMRI_Study_open --device cpu
```

---

## Troubleshooting

**Q: No visualizations generated?**

- A: Check that `./checkpoints_3d/training_history_3d.json` exists
- A: Ensure test data is in `./HipMRI_Study_open/semantic_MRs/`

**Q: Low Dice scores in visualization?**

- A: Check if class weights [0.25, 1.75] were applied
- A: Verify training ran for enough epochs (50+)

**Q: Out of memory?**

- A: Reduce `--num_detailed` parameter
- A: Use `--device cpu` instead of GPU

---

## Files Created/Modified

**New Functions** (visualization.py):

- `plot_training_curves()` - Generates loss+dice plot
- `visualize_3d_predictions_grid_10_per_image()` - 10 samples/PNG
- `visualize_3d_detailed_comparison()` - Multi-slice analysis
- `generate_all_3d_visualizations()` - Main orchestrator

**Modified Files**:

- `test3d.py` - Calls visualization after training

**Documentation**:

- `VISUALIZATION_GUIDE.md` - Complete user guide
- `VISUALIZATION_IMPLEMENTATION.md` - Technical details

---

## Key Metrics from Visualizations

### Training Curves Should Show:

- ✓ Both losses decreasing over epochs
- ✓ Validation loss stabilizing (not spiking)
- ✓ Dice score improving toward 0.9+
- ✓ Prostate Dice (class 1) reaching 0.85+

### Test Predictions Should Show:

- ✓ Red and blue mostly overlapping
- ✓ Consistent performance across batches
- ✓ Few false positives/negatives
- ✓ Clear prostate boundaries

### Detailed Analysis Should Show:

- ✓ Similar quality across all 5 slices
- ✓ Good performance on middle slices
- ✓ Edge slices sometimes less accurate (normal)
- ✓ No obvious artifacts or errors

---

## Next Steps

1. **Review visualizations**: Open `./results/training_curves_3d.png`
2. **Check test predictions**: Browse `./results/test_predictions_batch_*.png`
3. **Deep dive if needed**: Examine `./results/predictions_detailed/sample_*.png`
4. **If poor quality**: Check model training parameters and retry
5. **If good quality**: Ready for production/reporting

---

## Performance Benchmarks

| Component                      | Time       | Output Size |
| ------------------------------ | ---------- | ----------- |
| Training curves                | < 1 sec    | ~2-3 MB     |
| Test batches (100 samples)     | 5-10 sec   | ~50-80 MB   |
| Detailed analysis (10 samples) | 10-15 sec  | ~10-20 MB   |
| **Total**                      | ~30-40 sec | ~70-100 MB  |

---

## Command Reference

```bash
# Train + visualize (automatic)
python test3d.py

# Just visualizations (standalone)
python visualization.py --data_dir ./HipMRI_Study_open

# With custom options
python visualization.py \
  --checkpoint_dir ./checkpoints_3d \
  --data_dir ./HipMRI_Study_open \
  --num_detailed 10 \
  --downsample_factor 2 \
  --device auto
```

---

For more details, see:

- `VISUALIZATION_GUIDE.md` - Complete documentation
- `VISUALIZATION_IMPLEMENTATION.md` - Technical details
