import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from optimized_data_prep import get_data_loaders
from optimized_unet import UNet
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt
from pathlib import Path

class DiceLoss(nn.Module):
    """Dice loss for segmentation - applies sigmoid internally"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)  # Apply sigmoid to logits
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    """Combined BCE and Dice loss for better segmentation"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()  # Safe for autocast, includes sigmoid
        self.dice = DiceLoss()  # Applies sigmoid internally
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def calculate_iou(pred, target, threshold=0.5):
    """Calculate Intersection over Union"""
    pred = torch.sigmoid(pred)  # Apply sigmoid to logits
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

def calculate_dice(pred, target, threshold=0.5):
    """Calculate Dice coefficient"""
    pred = torch.sigmoid(pred)  # Apply sigmoid to logits
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    return dice.item()

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def plot_metrics(metrics, save_dir='./plots'):
    """Plot training metrics"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss plot
    axes[0, 0].plot(epochs, metrics['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, metrics['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # IoU plot
    axes[0, 1].plot(epochs, metrics['val_iou'], 'g-', label='Val IoU', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('IoU', fontsize=12)
    axes[0, 1].set_title('Validation IoU', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Dice plot
    axes[1, 0].plot(epochs, metrics['val_dice'], 'm-', label='Val Dice', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Dice Score', fontsize=12)
    axes[1, 0].set_title('Validation Dice Score', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1 plot
    axes[1, 1].plot(epochs, metrics['val_f1'], 'c-', label='Val F1', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('F1 Score', fontsize=12)
    axes[1, 1].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Metrics plot saved to {save_dir}/training_metrics.png")

def train_model(image_dir, mask_dir, num_epochs=100, batch_size=8, learning_rate=0.001,
                model_save_path='/content/best_model.pth'):
    """
    Optimized training with:
    - Mixed precision training (AMP)
    - Learning rate scheduling
    - Early stopping
    - Gradient clipping
    - Combined loss (BCE + Dice)
    - Comprehensive metrics (IoU, Dice, F1, Accuracy)
    """
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Initialize model
    model = UNet(n_channels=3, n_classes=1, bilinear=True, dropout=0.1).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function - Combined BCE and Dice
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    
    # Optimizer - AdamW with weight decay for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler - reduces LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15, min_delta=0.0001)
    
    # Mixed precision training for faster computation
    scaler = GradScaler('cuda')
    
    # Data loaders
    train_loader, val_loader = get_data_loaders(
        image_dir, mask_dir, batch_size=batch_size, num_workers=2
    )
    
    # Metrics tracking
    best_val_loss = float('inf')
    best_val_iou = 0.0
    metrics = {
        'train_loss': [], 'val_loss': [], 
        'val_iou': [], 'val_dice': [],
        'val_accuracy': [], 'val_f1': []
    }
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    for epoch in range(num_epochs):
        # ==================== TRAINING PHASE ====================
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [TRAIN]', 
                         leave=False, ncols=100)
        
        for batch_idx, (images, masks) in enumerate(train_pbar):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping for stable training
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # ==================== VALIDATION PHASE ====================
        model.eval()
        val_loss = 0
        all_preds = []
        all_masks = []
        all_iou = []
        all_dice = []
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [VAL]', 
                       leave=False, ncols=100)
        
        with torch.no_grad():
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)
                
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                
                # Calculate batch-wise IoU and Dice
                for i in range(outputs.size(0)):
                    pred = outputs[i]
                    mask = masks[i]
                    all_iou.append(calculate_iou(pred, mask))
                    all_dice.append(calculate_dice(pred, mask))
                
                # For pixel-wise accuracy and F1
                preds = torch.sigmoid(outputs)  # Apply sigmoid to logits
                preds = (preds > 0.5).float()
                all_preds.extend(preds.cpu().numpy().flatten())
                all_masks.extend(masks.cpu().numpy().flatten())
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate aggregated metrics
        val_accuracy = accuracy_score(all_masks, all_preds)
        val_f1 = f1_score(all_masks, all_preds, average='binary', zero_division=0)
        val_iou = np.mean(all_iou)
        val_dice = np.mean(all_dice)
        
        # Store metrics
        metrics['train_loss'].append(avg_train_loss)
        metrics['val_loss'].append(avg_val_loss)
        metrics['val_accuracy'].append(val_accuracy)
        metrics['val_f1'].append(val_f1)
        metrics['val_iou'].append(val_iou)
        metrics['val_dice'].append(val_dice)
        
        # Print epoch summary
        print(f'\n{"="*60}')
        print(f'EPOCH {epoch+1}/{num_epochs} SUMMARY')
        print(f'{"="*60}')
        print(f'Train Loss:      {avg_train_loss:.4f}')
        print(f'Val Loss:        {avg_val_loss:.4f}')
        print(f'Val Accuracy:    {val_accuracy:.4f} ({val_accuracy*100:.2f}%)')
        print(f'Val F1 Score:    {val_f1:.4f}')
        print(f'Val IoU:         {val_iou:.4f}')
        print(f'Val Dice:        {val_dice:.4f}')
        print(f'Learning Rate:   {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model based on IoU (most important metric for segmentation)
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_iou': val_iou,
                'val_dice': val_dice,
                'val_loss': avg_val_loss,
                'metrics': metrics
            }, model_save_path)
            print(f'✓ BEST MODEL SAVED! (IoU: {val_iou:.4f}, Dice: {val_dice:.4f})')
        
        print(f'{"="*60}\n')
        
        # Early stopping check
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"\n{'='*60}")
            print(f"EARLY STOPPING TRIGGERED AT EPOCH {epoch+1}")
            print(f"{'='*60}\n")
            break
    
    # Plot final metrics
    plot_metrics(metrics)
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Total Epochs:           {len(metrics['train_loss'])}")
    print(f"Best Validation IoU:    {best_val_iou:.4f}")
    print(f"Best Validation Dice:   {max(metrics['val_dice']):.4f}")
    print(f"Best Validation Loss:   {best_val_loss:.4f}")
    print(f"Model saved at:         {model_save_path}")
    print("="*60 + "\n")
    
    return model, metrics

if __name__ == '__main__':
    # ==================== CONFIGURATION ====================
    IMAGE_DIR = '/content/drive/My Drive/FYP Water data/images'
    MASK_DIR = '/content/drive/My Drive/FYP Water data/masks'
    MODEL_SAVE_PATH = '/content/best_model.pth'
    
    # Hyperparameters
    NUM_EPOCHS = 100
    BATCH_SIZE = 8  # Increase to 12-16 if you have more GPU memory
    LEARNING_RATE = 0.001
    
    print("="*60)
    print("U-NET WATER BODY SEGMENTATION")
    print("="*60)
    print(f"Configuration:")
    print(f"  Epochs:       {NUM_EPOCHS}")
    print(f"  Batch Size:   {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Image Dir:    {IMAGE_DIR}")
    print(f"  Mask Dir:     {MASK_DIR}")
    print("="*60 + "\n")
    
    # Verify directories exist
    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"Images directory not found: {IMAGE_DIR}")
    if not os.path.exists(MASK_DIR):
        raise FileNotFoundError(f"Masks directory not found: {MASK_DIR}")
    
    try:
        model, metrics = train_model(
            IMAGE_DIR, 
            MASK_DIR, 
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            model_save_path=MODEL_SAVE_PATH
        )
        
        print("\n✓ Training completed successfully!")
        print(f"✓ Final metrics:")
        print(f"  - Best IoU: {max(metrics['val_iou']):.4f}")
        print(f"  - Best Dice: {max(metrics['val_dice']):.4f}")
        print(f"  - Best F1: {max(metrics['val_f1']):.4f}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise