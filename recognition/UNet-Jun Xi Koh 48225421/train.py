"""
Unified training script for 2D and 3D Improved UNet segmentation.

This script:
- Trains the Improved UNet model on 2D slices OR 3D volumes
- Validates on validation set
- Tests on test set
- Tracks Dice coefficient metric
- Saves the best model checkpoint
- Supports both 2D and 3D modes with automatic model selection

Usage:
    2D: python train.py --mode 2d --data_dir ./HipMRI_Study_open/keras_slices_data --epochs 50
    3D: python train.py --mode 3d --data_dir ./HipMRI_Study_open --epochs 50 --downsample_factor 2
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from modules import ImprovedUNet2D, ImprovedUNet3D, DiceLoss, dice_coefficient, dice_coefficient_3d
from dataset import create_data_loaders


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for model parameters
        criterion: Loss function
        device: Device to train on (cuda or cpu)
        
    Returns:
        Average loss for the epoch
    """
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    with tqdm(train_loader, desc="Training") as pbar:
        for batch_data in pbar:
            images, labels = batch_data
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate_epoch(model, val_loader, criterion, device, mode='2d', num_classes=2):
    """
    Validate the model on validation set.
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        mode: '2d' or '3d'
        num_classes: Number of classes for Dice calculation
        
    Returns:
        Tuple of (average loss, average Dice coefficient)
    """
    
    model.eval()
    total_loss = 0.0
    total_dice = {f'class_{c}': 0.0 for c in range(num_classes)}
    total_dice['avg'] = 0.0
    num_batches = 0
    
    dice_fn = dice_coefficient_3d if mode == '3d' else dice_coefficient
    
    with torch.no_grad():
        with tqdm(val_loader, desc="Validating") as pbar:
            for batch_data in pbar:
                images, labels = batch_data
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Dice coefficient
                dice = dice_fn(outputs, labels, num_classes=num_classes)
                
                # Track metrics
                total_loss += loss.item()
                for key in total_dice.keys():
                    total_dice[key] += dice[key]
                num_batches += 1
                
                pbar.set_postfix({'loss': loss.item(), 'dice': dice['avg']})
    
    # Average metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_dice = {}
    for key in total_dice.keys():
        avg_dice[key] = total_dice[key] / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_dice


def test_epoch(model, test_loader, device, mode='2d', num_classes=2):
    """
    Test the model on test set.
    
    Args:
        model: The model to test
        test_loader: DataLoader for test data
        device: Device to test on
        mode: '2d' or '3d'
        num_classes: Number of classes for Dice calculation
        
    Returns:
        Dictionary with Dice scores for each class on test set
    """
    
    model.eval()
    total_dice = {f'class_{c}': 0.0 for c in range(num_classes)}
    total_dice['avg'] = 0.0
    num_batches = 0
    
    dice_fn = dice_coefficient_3d if mode == '3d' else dice_coefficient
    
    with torch.no_grad():
        with tqdm(test_loader, desc="Testing") as pbar:
            for batch_data in pbar:
                images, labels = batch_data
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Dice coefficient
                dice = dice_fn(outputs, labels, num_classes=num_classes)
                
                # Track metrics
                for key in total_dice.keys():
                    total_dice[key] += dice[key]
                num_batches += 1
                
                pbar.set_postfix({'dice': dice['avg']})
    
    # Average metrics
    avg_dice = {}
    for key in total_dice.keys():
        avg_dice[key] = total_dice[key] / num_batches if num_batches > 0 else 0.0
    
    return avg_dice


def train(data_dir, mode='2d', num_epochs=50, batch_size=1, learning_rate=1e-3, 
          num_classes=2, device=None, checkpoint_dir="./checkpoints", 
          early_stopping_patience=10, downsample_factor=2):
    """
    Train the Improved UNet model (2D or 3D).
    
    Args:
        data_dir: Path to data directory
        mode: '2d' or '3d' (default: '2d')
        num_epochs: Number of training epochs (default: 50)
        batch_size: Batch size (default: 1)
        learning_rate: Learning rate (default: 1e-3)
        num_classes: Number of classes (default: 2)
        device: Device to train on (default: auto-detect)
        checkpoint_dir: Directory to save checkpoints (default: ./checkpoints)
        early_stopping_patience: Patience for early stopping (default: 10)
        downsample_factor: Downsampling factor for 3D (default: 2)
        
    Returns:
        Training history dictionary
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading {mode.upper()} dataset...")
    if mode.lower() == '3d':
        data_loaders = create_data_loaders(
            data_dir, 
            batch_size=batch_size, 
            num_workers=0, 
            mode='3d',
            downsample_factor=downsample_factor
        )
        train_loader = data_loaders['train_loader']
        val_loader = data_loaders['val_loader']
        test_loader = data_loaders['test_loader']
    else:
        train_loader, val_loader, test_loader, _, _, _ = create_data_loaders(
            data_dir, 
            batch_size=batch_size, 
            num_workers=4, 
            mode='2d'
        )
    
    # Initialize model
    print(f"\nInitializing {mode.upper()} UNet model...")
    if mode.lower() == '3d':
        model = ImprovedUNet3D(
            in_channels=1,
            num_classes=num_classes,
            filters=32,  # Smaller for 3D to save memory
            dropout_rate=0.5
        ).to(device)
    else:
        model = ImprovedUNet2D(
            in_channels=1,
            num_classes=num_classes,
            filters=64,
            dropout_rate=0.5
        ).to(device)
    
    # Loss and optimizer
    # Class weights to handle class imbalance (inversely proportional to class frequency)
    class_weights = [0.25, 1.75] if mode.lower() == '3d' else None  # 3D has severe imbalance
    criterion = DiceLoss(smooth=1.0, num_classes=num_classes, class_weights=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'best_epoch': 0,
        'best_val_dice': 0.0,
        'test_dice': None,
        'mode': mode,
        'downsample_factor': downsample_factor if mode.lower() == '3d' else None
    }
    
    best_val_dice = 0.0
    patience_counter = 0
    
    print(f"\nTraining {mode.upper()} UNet...")
    print("="*60)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)
        
        # Validate
        val_loss, val_dice = validate_epoch(model, val_loader, criterion, device, mode=mode, num_classes=num_classes)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Val Dice - Avg: {val_dice['avg']:.4f}", end="")
        for c in range(num_classes):
            print(f", Class {c}: {val_dice[f'class_{c}']:.4f}", end="")
        print()
        
        # Learning rate scheduling
        scheduler.step(val_dice['avg'])
        
        # Early stopping and checkpointing
        if val_dice['avg'] > best_val_dice:
            best_val_dice = val_dice['avg']
            history['best_val_dice'] = best_val_dice
            history['best_epoch'] = epoch
            patience_counter = 0
            
            # Save best model
            checkpoint_path = checkpoint_dir / f"best_model_epoch_{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Test
    print("\n" + "="*60)
    print("TESTING")
    print("="*60)
    
    # Load best model
    best_model_path = checkpoint_dir / f"best_model_epoch_{history['best_epoch']}.pt"
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    # Test evaluation
    test_dice = test_epoch(model, test_loader, device, mode=mode, num_classes=num_classes)
    history['test_dice'] = test_dice
    
    print(f"\nTest Dice - Avg: {test_dice['avg']:.4f}", end="")
    for c in range(num_classes):
        print(f", Class {c}: {test_dice[f'class_{c}']:.4f}", end="")
    print()
    
    # Save training history
    history_path = checkpoint_dir / f"training_history_{mode}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    
    return history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Improved UNet for medical image segmentation')
    parser.add_argument('--mode', type=str, default='2d', choices=['2d', '3d'], 
                        help='Training mode: 2d or 3d')
    parser.add_argument('--data_dir', type=str, default='./HipMRI_Study_open',
                        help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--downsample_factor', type=int, default=2,
                        help='Downsampling factor for 3D (only used in 3d mode)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Adjust checkpoint directory based on mode
    checkpoint_dir = f"{args.checkpoint_dir}_{args.mode}" if args.mode == '3d' else args.checkpoint_dir
    
    history = train(
        data_dir=args.data_dir,
        mode=args.mode,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_classes=2,
        device=device,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=10,
        downsample_factor=args.downsample_factor
    )
