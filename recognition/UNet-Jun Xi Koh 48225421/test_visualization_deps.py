"""
Quick test to verify visualization module imports correctly.
Run this to ensure all dependencies are available.
"""

import sys
from pathlib import Path

print("Testing visualization module dependencies...")
print("=" * 60)

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib.pyplot")
except ImportError as e:
    print(f"✗ matplotlib.pyplot: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("✓ numpy")
except ImportError as e:
    print(f"✗ numpy: {e}")
    sys.exit(1)

try:
    import json
    print("✓ json")
except ImportError as e:
    print(f"✗ json: {e}")
    sys.exit(1)

try:
    from visualization import (
        plot_dice_loss_curves,
        plot_test_predictions,
        plot_test_grid,
        plot_dice_distribution,
        create_visualization_summary
    )
    print("✓ visualization module")
except ImportError as e:
    print(f"✗ visualization module: {e}")
    sys.exit(1)

print("=" * 60)
print("✓ All dependencies available!")
print("\nYou can now run:")
print("  - python train.py --data_dir <path_to_data>")
print("  - python predict.py --checkpoint <model_path> --data_dir <path_to_data>")
print("\nVisualizations will be automatically saved!")
