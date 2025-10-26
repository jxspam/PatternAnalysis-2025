"""
Improved UNet architectures for medical image segmentation.

This module contains:
- 2D Improved UNet for 2D slice segmentation
- 3D Improved UNet for volumetric segmentation
- Batch normalization for stable training
- Dropout layers for regularization
- Skip connections for better gradient flow
- Shared DiceLoss and evaluation metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Convolutional block with two conv layers, batch normalization, and ReLU activation.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolutional kernel (default: 3)
        padding: Padding for convolution (default: 1)
        dropout_rate: Dropout rate for regularization (default: 0.5)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.5):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                               padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, 
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(dropout_rate)
    
    def forward(self, x):
        """Forward pass through the conv block."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        return x


class UpConvBlock(nn.Module):
    """
    Upsampling block with transpose convolution.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """
    
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        """Forward pass through the upsampling block."""
        return self.up(x)


class ImprovedUNet2D(nn.Module):
    """
    Improved 2D UNet for medical image segmentation with:
    - Batch normalization for stable training
    - Dropout for regularization
    - Skip connections with concatenation
    - Optimized for segmentation tasks
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale)
        num_classes: Number of output classes (default: 2 for binary segmentation)
        filters: Base number of filters (default: 64)
        dropout_rate: Dropout rate (default: 0.5)
    """
    
    def __init__(self, in_channels=1, num_classes=2, filters=64, dropout_rate=0.5):
        super(ImprovedUNet2D, self).__init__()
        
        self.filters = filters
        self.dropout_rate = dropout_rate
        
        # Encoder (Downsampling path)
        self.enc1 = ConvBlock(in_channels, filters, dropout_rate=dropout_rate)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = ConvBlock(filters, filters * 2, dropout_rate=dropout_rate)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = ConvBlock(filters * 2, filters * 4, dropout_rate=dropout_rate)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = ConvBlock(filters * 4, filters * 8, dropout_rate=dropout_rate)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(filters * 8, filters * 16, dropout_rate=dropout_rate)
        
        # Decoder (Upsampling path)
        self.up4 = UpConvBlock(filters * 16, filters * 8)
        self.dec4 = ConvBlock(filters * 16, filters * 8, dropout_rate=dropout_rate)
        
        self.up3 = UpConvBlock(filters * 8, filters * 4)
        self.dec3 = ConvBlock(filters * 8, filters * 4, dropout_rate=dropout_rate)
        
        self.up2 = UpConvBlock(filters * 4, filters * 2)
        self.dec2 = ConvBlock(filters * 4, filters * 2, dropout_rate=dropout_rate)
        
        self.up1 = UpConvBlock(filters * 2, filters)
        self.dec1 = ConvBlock(filters * 2, filters, dropout_rate=dropout_rate)
        
        # Output layer
        self.out = nn.Conv2d(filters, num_classes, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass through the Improved UNet.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Output tensor of shape (batch, num_classes, height, width)
        """
        
        # Encoder with skip connections
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        
        enc4 = self.enc4(pool3)
        pool4 = self.pool4(enc4)
        
        # Bottleneck
        bottleneck = self.bottleneck(pool4)
        
        # Decoder with skip connections (concatenation)
        up4 = self.up4(bottleneck)
        # Handle potential size mismatches due to odd dimensions
        if up4.shape != enc4.shape:
            up4 = F.interpolate(up4, size=enc4.shape[2:], mode='bilinear', align_corners=False)
        dec4_input = torch.cat([up4, enc4], dim=1)
        dec4 = self.dec4(dec4_input)
        
        up3 = self.up3(dec4)
        if up3.shape != enc3.shape:
            up3 = F.interpolate(up3, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec3_input = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec3(dec3_input)
        
        up2 = self.up2(dec3)
        if up2.shape != enc2.shape:
            up2 = F.interpolate(up2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2_input = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(dec2_input)
        
        up1 = self.up1(dec2)
        if up1.shape != enc1.shape:
            up1 = F.interpolate(up1, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1_input = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(dec1_input)
        
        # Output
        out = self.out(dec1)
        
        return out


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation with optional class weighting.
    
    The Dice coefficient is calculated as: 2 * (intersection) / (sum of areas)
    Loss = 1 - Dice coefficient
    
    Args:
        smooth: Smoothing constant to avoid division by zero (default: 1.0)
        num_classes: Number of classes (default: 2)
        class_weights: Optional list of weights for each class to handle class imbalance.
                      If None, all classes are weighted equally. (default: None)
    """
    
    def __init__(self, smooth=1.0, num_classes=2, class_weights=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        # Use provided weights or default to equal weights
        if class_weights is None:
            self.class_weights = [1.0] * num_classes
        else:
            self.class_weights = class_weights
    
    def forward(self, predictions, targets):
        """
        Calculate weighted Dice loss.
        
        Args:
            predictions: Model output of shape (batch, num_classes, height, width) or 3D
            targets: Ground truth labels of shape (batch, height, width) or 3D
            
        Returns:
            Weighted Dice loss value
        """
        
        # Convert predictions to probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot if needed (handles both 2D and 3D)
        if targets.dim() == 3 or targets.dim() == 4:
            targets_one_hot = torch.zeros_like(predictions)
            for c in range(self.num_classes):
                targets_one_hot[:, c] = (targets == c).float()
        else:
            targets_one_hot = targets
        
        # Calculate weighted Dice loss for each class
        dice_losses = []
        for c in range(self.num_classes):
            pred_c = predictions[:, c]
            target_c = targets_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum()
            pred_sum = pred_c.sum()
            target_sum = target_c.sum()
            
            dice = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
            # Apply class weight to loss
            weighted_loss = (1 - dice) * self.class_weights[c]
            dice_losses.append(weighted_loss)
        
        # Return average Dice loss across all classes
        return torch.stack(dice_losses).mean()


def dice_coefficient(predictions, targets, num_classes=2, smooth=1.0):
    """
    Calculate Dice coefficient for evaluation.
    
    Args:
        predictions: Model output probabilities of shape (batch, num_classes, height, width)
        targets: Ground truth labels of shape (batch, height, width) or one-hot encoded
        num_classes: Number of classes
        smooth: Smoothing constant
        
    Returns:
        Dictionary with Dice scores for each class and average Dice score
    """
    
    # Convert predictions to class indices
    pred_classes = torch.argmax(predictions, dim=1)
    
    dice_scores = {}
    
    for c in range(num_classes):
        pred_c = (pred_classes == c).float()
        
        if targets.dim() == 3:
            target_c = (targets == c).float()
        else:
            target_c = targets[:, c]
        
        intersection = (pred_c * target_c).sum()
        pred_sum = pred_c.sum()
        target_sum = target_c.sum()
        
        dice = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
        dice_scores[f'class_{c}'] = dice.item()
    
    # Average Dice score
    avg_dice = sum(dice_scores.values()) / num_classes
    dice_scores['avg'] = avg_dice
    
    return dice_scores


# ========== 3D UNet Components ==========

class ConvBlock3D(nn.Module):
    """
    3D Convolutional block with two conv layers, batch normalization, and ReLU activation.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolutional kernel (default: 3)
        padding: Padding for convolution (default: 1)
        dropout_rate: Dropout rate for regularization (default: 0.5)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.5):
        super(ConvBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                               padding=padding, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout3d(dropout_rate)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, 
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout3d(dropout_rate)
    
    def forward(self, x):
        """Forward pass through the 3D conv block."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        return x


class UpConvBlock3D(nn.Module):
    """
    3D Upsampling block with transpose convolution.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """
    
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock3D, self).__init__()
        
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        """Forward pass through the 3D upsampling block."""
        return self.up(x)


class ImprovedUNet3D(nn.Module):
    """
    Improved 3D UNet for volumetric medical image segmentation with:
    - Batch normalization for stable training
    - Dropout for regularization
    - Skip connections with concatenation
    - Optimized for 3D segmentation tasks
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale)
        num_classes: Number of output classes (default: 2 for binary segmentation)
        filters: Base number of filters (default: 32 for 3D to save memory)
        dropout_rate: Dropout rate (default: 0.5)
    """
    
    def __init__(self, in_channels=1, num_classes=2, filters=32, dropout_rate=0.5):
        super(ImprovedUNet3D, self).__init__()
        
        self.filters = filters
        self.dropout_rate = dropout_rate
        
        # Encoder (Downsampling path) - 3 levels for 3D
        self.enc1 = ConvBlock3D(in_channels, filters, dropout_rate=dropout_rate)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = ConvBlock3D(filters, filters * 2, dropout_rate=dropout_rate)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc3 = ConvBlock3D(filters * 2, filters * 4, dropout_rate=dropout_rate)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck (deepest level)
        self.bottleneck = ConvBlock3D(filters * 4, filters * 8, dropout_rate=dropout_rate)
        
        # Decoder (Upsampling path)
        self.up3 = UpConvBlock3D(filters * 8, filters * 4)
        self.dec3 = ConvBlock3D(filters * 8, filters * 4, dropout_rate=dropout_rate)
        
        self.up2 = UpConvBlock3D(filters * 4, filters * 2)
        self.dec2 = ConvBlock3D(filters * 4, filters * 2, dropout_rate=dropout_rate)
        
        self.up1 = UpConvBlock3D(filters * 2, filters)
        self.dec1 = ConvBlock3D(filters * 2, filters, dropout_rate=dropout_rate)
        
        # Output layer
        self.out = nn.Conv3d(filters, num_classes, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass through the Improved 3D UNet.
        
        Args:
            x: Input tensor of shape (batch, channels, depth, height, width)
            
        Returns:
            Output tensor of shape (batch, num_classes, depth, height, width)
        """
        
        # Encoder with skip connections
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(pool3)
        
        # Decoder with skip connections (concatenation)
        up3 = self.up3(bottleneck)
        # Handle potential size mismatches due to odd dimensions
        if up3.shape != enc3.shape:
            up3 = F.interpolate(up3, size=enc3.shape[2:], mode='trilinear', align_corners=False)
        dec3_input = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec3(dec3_input)
        
        up2 = self.up2(dec3)
        if up2.shape != enc2.shape:
            up2 = F.interpolate(up2, size=enc2.shape[2:], mode='trilinear', align_corners=False)
        dec2_input = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(dec2_input)
        
        up1 = self.up1(dec2)
        if up1.shape != enc1.shape:
            up1 = F.interpolate(up1, size=enc1.shape[2:], mode='trilinear', align_corners=False)
        dec1_input = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(dec1_input)
        
        # Output
        out = self.out(dec1)
        
        return out


def dice_coefficient_3d(predictions, targets, num_classes=2, smooth=1.0):
    """
    Calculate Dice coefficient for 3D segmentation evaluation.
    
    Args:
        predictions: Model output probabilities of shape (batch, num_classes, depth, height, width)
        targets: Ground truth labels of shape (batch, depth, height, width)
        num_classes: Number of classes
        smooth: Smoothing constant
        
    Returns:
        Dictionary with Dice scores for each class and average Dice score
    """
    
    # Convert predictions to class indices
    pred_classes = torch.argmax(predictions, dim=1)
    
    dice_scores = {}
    
    for c in range(num_classes):
        pred_c = (pred_classes == c).float()
        target_c = (targets == c).float()
        
        intersection = (pred_c * target_c).sum()
        pred_sum = pred_c.sum()
        target_sum = target_c.sum()
        
        dice = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
        dice_scores[f'class_{c}'] = dice.item()
    
    # Average Dice score
    avg_dice = sum(dice_scores.values()) / num_classes
    dice_scores['avg'] = avg_dice
    
    return dice_scores

