"""
Training script for 3D Improved UNet volumetric segmentation.

This script:
- Trains the Improved 3D UNet model on volumetric data
- Validates on validation set
- Tests on test set
- Tracks Dice coefficient metric per class
- Saves the best model checkpoint
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

from modules import ImprovedUNet3D, DiceLoss, dice_coefficient_3d
from dataset3d import create_3d_data_loaders


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train for one epoch on 3D volumetric data.
    
    Args:
        model: The 3D model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for model parameters
        criterion: Loss function (Dice Loss)
        device: Device to train on (cuda or cpu)
        
    Returns:
        Average loss for the epoch
    """
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    with tqdm(train_loader, desc="Training") as pbar:
        for volumes, labels in pbar:
            volumes = volumes.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(volumes)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate_epoch(model, val_loader, criterion, device, num_classes=2):
    """
    Validate the model on validation set.
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        num_classes: Number of classes for Dice calculation
        
    Returns:
        Tuple of (average loss, average Dice coefficient)
    """
    
    model.eval()
    total_loss = 0.0
    total_dice = {f'class_{c}': 0.0 for c in range(num_classes)}
    total_dice['avg'] = 0.0
    num_batches = 0
    
    with torch.no_grad():
        with tqdm(val_loader, desc="Validating") as pbar:
            for volumes, labels in pbar:
                volumes = volumes.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(volumes)
                loss = criterion(outputs, labels)
                
                # Dice coefficient
                dice = dice_coefficient_3d(outputs, labels, num_classes=num_classes)
                
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


def test_epoch(model, test_loader, device, num_classes=2):
    """
    Test the model on test set.
    
    Args:
        model: The model to test
        test_loader: DataLoader for test data
        device: Device to test on
        num_classes: Number of classes for Dice calculation
        
    Returns:
        Dictionary with Dice scores for each class on test set
    """
    
    model.eval()
    total_dice = {f'class_{c}': 0.0 for c in range(num_classes)}
    total_dice['avg'] = 0.0
    num_batches = 0
    
    with torch.no_grad():
        with tqdm(test_loader, desc="Testing") as pbar:
            for volumes, labels in pbar:
                volumes = volumes.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(volumes)
                
                # Dice coefficient
                dice = dice_coefficient_3d(outputs, labels, num_classes=num_classes)
                
                # Track metrics
                for key in total_dice.keys():
                    total_dice[key] += dice[key]
                num_batches += 1
                
                pbar.set_postfix({'dice': dice['avg']})
    
    # Average metrics
    for key in total_dice.keys():
        total_dice[key] = total_dice[key] / num_batches if num_batches > 0 else 0.0
    
    return total_dice


def train(data_dir, num_epochs=1, batch_size=1, learning_rate=1e-3, num_classes=2,
          device=None, checkpoint_dir="./checkpoints", early_stopping_patience=2,
          downsample_factor=2):
    """
    Full training loop for 3D UNet.
    
    Args:
        data_dir: Path to data directory with semantic_MRs and semantic_labels_only
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for optimizer
        num_classes: Number of segmentation classes
        device: Device to train on (cuda or cpu)
        checkpoint_dir: Directory to save checkpoints
        early_stopping_patience: Patience for early stopping
        downsample_factor: Downsampling factor for 3D volumes
        
    Returns:
        Dictionary with training history
    """
    
    # Default device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    print("\nLoading 3D data...")
    data_loaders = create_3d_data_loaders(
        data_dir,
        batch_size=batch_size,
        num_workers=0,  # No workers for 3D due to memory
        downsample_factor=downsample_factor
    )
    
    train_loader = data_loaders['train_loader']
    val_loader = data_loaders['val_loader']
    test_loader = data_loaders['test_loader']
    
    # Initialize model
    print("\nInitializing 3D UNet model...")
    model = ImprovedUNet3D(
        in_channels=1,
        num_classes=num_classes,
        filters=32,  # Smaller for 3D to save memory
        dropout_rate=0.5
    ).to(device)
    
    # Loss and optimizer
    criterion = DiceLoss(smooth=1.0, num_classes=num_classes)
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
        'downsample_factor': downsample_factor
    }
    
    best_val_dice = 0.0
    patience_counter = 0
    
    print("\nTraining 3D UNet...")
    print("="*60)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)
        
        # Validate
        val_loss, val_dice = validate_epoch(
            model, val_loader, criterion, device, num_classes=num_classes
        )
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
    test_dice = test_epoch(model, test_loader, device, num_classes=num_classes)
    history['test_dice'] = test_dice
    
    print(f"\nTest Dice - Avg: {test_dice['avg']:.4f}", end="")
    for c in range(num_classes):
        print(f", Class {c}: {test_dice[f'class_{c}']:.4f}", end="")
    print()
    
    # Save training history
    history_path = checkpoint_dir / "training_history_3d.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    
    return history


if __name__ == "__main__":
    # Default parameters
    data_dir = "./HipMRI_Study_open"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    history = train(
        data_dir=data_dir,
        num_epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        num_classes=2,
        device=device,
        checkpoint_dir="./checkpoints_3d",
        early_stopping_patience=2,
        downsample_factor=2
    )
