import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class MaskDataset(Dataset):
    """
    Optimized dataset with:
    - Albumentations for better augmentation
    - Proper image preprocessing
    - Data validation
    """
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(256, 256), augment=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.augment = augment
        
        # Get all image files
        self.images = sorted([f for f in os.listdir(image_dir) 
                            if f.startswith('s') and f.endswith('.jpg')])
        print(f"Found {len(self.images)} images in {image_dir}")
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        # Verify corresponding masks exist
        self.valid_pairs = []
        for img_name in self.images:
            mask_name = img_name.replace('.jpg', '_mask.png')
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                self.valid_pairs.append((img_name, mask_name))
            else:
                print(f"Warning: No mask found for {img_name}")
        
        if len(self.valid_pairs) == 0:
            raise ValueError(f"No valid image-mask pairs found")
            
        print(f"Found {len(self.valid_pairs)} valid image-mask pairs")
        
        # Setup transforms
        if augment:
            self.transform = A.Compose([
                A.Resize(target_size[0], target_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    scale=(0.9, 1.1), 
                    translate_percent=(-0.1, 0.1), 
                    rotate=(-15, 15), 
                    shear=(-5, 5), 
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.5
                ),
                A.GaussNoise(p=0.3),
                A.Blur(blur_limit=3, p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(target_size[0], target_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        img_name, mask_name = self.valid_pairs[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Read with cv2 for better performance
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Normalize mask to 0-1 (binary)
        mask = (mask > 127).astype(np.float32)
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Ensure mask is in correct shape [1, H, W]
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        return image, mask.float()

def get_data_loaders(image_dir, mask_dir, batch_size=8, train_split=0.8, 
                     num_workers=2, pin_memory=True):
    """
    Create optimized data loaders with:
    - Proper train/val split with seed
    - Multi-worker loading
    - Pin memory for GPU
    """
    image_dir = os.path.abspath(image_dir)
    mask_dir = os.path.abspath(mask_dir)
    
    print(f"Loading data from:")
    print(f"Images: {image_dir}")
    print(f"Masks: {mask_dir}")
    
    # Create datasets
    train_dataset = MaskDataset(image_dir, mask_dir, augment=True)
    val_dataset = MaskDataset(image_dir, mask_dir, augment=False)
    
    # Split indices
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(train_split * dataset_size))
    
    # Set seed for reproducibility
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices, val_indices = indices[:split], indices[split:]
    
    # Create samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True  # Drop last incomplete batch for consistent training
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    
    return train_loader, val_loader