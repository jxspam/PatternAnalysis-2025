"""
Training script for 2D Improved UNet segmentation.

This script:
- Trains the Improved UNet model on 2D slices
- Validates on validation set
- Tests on test set
- Tracks Dice coefficient metric
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

from modules import ImprovedUNet2D, DiceLoss, dice_coefficient
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
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(images)
            
            # Calculate loss
            loss = criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches


def validate_epoch(model, val_loader, criterion, device, num_classes=2):
    """
    Validate the model on validation set.
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        num_classes: Number of classes
        
    Returns:
        Tuple of (average loss, average Dice score)
    """
    
    model.eval()
    total_loss = 0.0
    all_dice_scores = {f'class_{i}': [] for i in range(num_classes)}
    all_dice_scores['avg'] = []
    num_batches = 0
    
    with torch.no_grad():
        with tqdm(val_loader, desc="Validating") as pbar:
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                predictions = model(images)
                
                # Calculate loss
                loss = criterion(predictions, labels)
                total_loss += loss.item()
                
                # Calculate Dice coefficient
                dice_scores = dice_coefficient(predictions, labels, num_classes=num_classes)
                for key in dice_scores:
                    all_dice_scores[key].append(dice_scores[key])
                
                num_batches += 1
                pbar.set_postfix({
                    'loss': loss.item(),
                    'avg_dice': dice_scores['avg']
                })
    
    avg_loss = total_loss / num_batches
    avg_dice_scores = {key: np.mean(values) for key, values in all_dice_scores.items()}
    
    return avg_loss, avg_dice_scores


def test_epoch(model, test_loader, device, num_classes=2):
    """
    Test the model on test set.
    
    Args:
        model: The model to test
        test_loader: DataLoader for test data
        device: Device to test on
        num_classes: Number of classes
        
    Returns:
        Dictionary with Dice scores for each class
    """
    
    model.eval()
    all_dice_scores = {f'class_{i}': [] for i in range(num_classes)}
    all_dice_scores['avg'] = []
    
    with torch.no_grad():
        with tqdm(test_loader, desc="Testing") as pbar:
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                predictions = model(images)
                
                # Calculate Dice coefficient
                dice_scores = dice_coefficient(predictions, labels, num_classes=num_classes)
                for key in dice_scores:
                    all_dice_scores[key].append(dice_scores[key])
                
                pbar.set_postfix({'avg_dice': dice_scores['avg']})
    
    test_dice_scores = {key: np.mean(values) for key, values in all_dice_scores.items()}
    
    return test_dice_scores


def train(
    data_dir,
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    num_classes=2,
    device=None,
    checkpoint_dir='./checkpoints',
    early_stopping_patience=15
):
    """
    Train the Improved UNet model.
    
    Args:
        data_dir: Path to keras_slices_data directory
        num_epochs: Number of epochs to train (default: 100)
        batch_size: Batch size (default: 32)
        learning_rate: Learning rate (default: 1e-3)
        num_classes: Number of classes (default: 2)
        device: Device to train on (default: cuda if available, else cpu)
        checkpoint_dir: Directory to save checkpoints (default: './checkpoints')
        early_stopping_patience: Patience for early stopping (default: 15)
        
    Returns:
        Dictionary with training history
    """
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    print(f"Loading data from {data_dir}")
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = \
        create_data_loaders(data_dir, batch_size=batch_size, num_workers=0)
    
    # Create model
    print("Creating Improved UNet2D model...")
    model = ImprovedUNet2D(in_channels=1, num_classes=num_classes, filters=64, dropout_rate=0.5)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = DiceLoss(smooth=1.0, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'test_dice': None,
        'best_epoch': 0,
        'best_val_dice': 0.0
    }
    
    best_val_dice = 0.0
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)
        print(f"Train Loss: {train_loss:.6f}")
        
        # Validate
        val_loss, val_dice = validate_epoch(model, val_loader, criterion, device, num_classes)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Val Dice Scores: {val_dice}")
        
        # Learning rate scheduler
        scheduler.step(val_dice['avg'])
        
        # Save best model
        if val_dice['avg'] > best_val_dice:
            best_val_dice = val_dice['avg']
            patience_counter = 0
            history['best_epoch'] = epoch + 1
            history['best_val_dice'] = best_val_dice
            
            # Save checkpoint
            checkpoint_path = checkpoint_dir / f"best_model_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_dice': best_val_dice,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (patience: {patience_counter})")
            break
    
    # Load best model and test
    print("\n" + "="*50)
    print("Loading best model and testing...")
    best_checkpoint_path = checkpoint_dir / f"best_model_epoch_{history['best_epoch']}.pt"
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    test_dice = test_epoch(model, test_loader, device, num_classes)
    history['test_dice'] = test_dice
    print(f"Test Dice Scores: {test_dice}")
    
    # Plot training history
    plot_training_history(history, checkpoint_dir)
    
    # Save history
    history_path = checkpoint_dir / "training_history.json"
    # Convert numpy arrays to lists for JSON serialization
    history_json = {
        'train_loss': [float(x) for x in history['train_loss']],
        'val_loss': [float(x) for x in history['val_loss']],
        'val_dice': history['val_dice'],
        'test_dice': history['test_dice'],
        'best_epoch': history['best_epoch'],
        'best_val_dice': float(history['best_val_dice'])
    }
    with open(history_path, 'w') as f:
        json.dump(history_json, f, indent=4)
    print(f"\nSaved training history to {history_path}")
    
    return history


def plot_training_history(history, output_dir):
    """
    Plot and save training history.
    
    Args:
        history: Dictionary with training history
        output_dir: Directory to save plots
    """
    
    output_dir = Path(output_dir)
    
    # Plot loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Dice plot
    val_dice_avg = [d['avg'] for d in history['val_dice']]
    ax2.plot(val_dice_avg, label='Val Dice (Avg)', linewidth=2, color='green')
    ax2.axhline(y=0.75, color='red', linestyle='--', label='Target Dice (0.75)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Coefficient')
    ax2.set_title('Validation Dice Coefficient')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'training_history.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Improved UNet for 2D segmentation')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to keras_slices_data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Train
    history = train(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_classes=args.num_classes,
        checkpoint_dir=args.checkpoint_dir,
        early_stopping_patience=args.early_stopping_patience
    )
