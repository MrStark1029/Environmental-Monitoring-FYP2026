import torch
import cv2
import numpy as np
from PIL import Image
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from optimized_unet import UNet

def apply_mask(image, mask):
    """Apply binary mask to image"""
    image_np = np.array(image)
    mask_np = np.array(mask)
    # Expand mask to 3 channels
    mask_3channel = np.stack([mask_np] * 3, axis=2) / 255
    # Apply mask
    masked_image = (image_np * mask_3channel).astype(np.uint8)
    return Image.fromarray(masked_image)

def predict_mask(image_path, model_path='best_model.pth', save_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model with updated loading logic
    try:
        # First try loading with weights_only
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model = UNet(n_channels=3, n_classes=1).to(device)
        model.load_state_dict(checkpoint)
    except Exception as e:
        print("Attempting to load model with weights_only=False...")
        try:
            # If that fails, try loading without weights_only (for older checkpoints)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model = UNet(n_channels=3, n_classes=1).to(device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    model.eval()
    
    # Load image with cv2 for consistency with training
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # Use same transforms as validation
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Apply transforms
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Predict mask
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            output = model(image_tensor)
            mask = torch.sigmoid(output) > 0.5
            mask = mask.squeeze().cpu().numpy()
    
    # Resize mask back to original size
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    mask = Image.fromarray(mask)
    
    # Create output directory if provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Convert image back to PIL for consistency
    original_image = Image.fromarray(image)
    
    # Apply mask to original image
    masked_image = apply_mask(original_image, mask)
    
    # Save outputs if directory provided
    if save_dir:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        mask.save(os.path.join(save_dir, f"{base_name}_mask.png"))
        masked_image.save(os.path.join(save_dir, f"{base_name}_masked.png"))
    
    return mask, masked_image


if __name__ == '__main__':
    image_path = 'images/s143.jpg'
    save_dir = 'predictions'
    mask, masked_image = predict_mask(image_path, save_dir=save_dir)
    print(f"Saved predictions to {save_dir}")
