"""
Image augmentatio piplines using Albumentations.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_train_transforms(img_size: int = 224) -> A.Compose:
    """
    Get training image transformations with augmentations.

    Args:
        img_size: Target image size.
    
    Returns:
        ALbumentations composition
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            p=0.5
        ),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),

        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        ], p=0.3),
        A.CoarseDropout(
            max_holes=8,
            max_height=img_size // 8,
            max_width=img_size // 8,
            fill_value=0,
            p=0.3
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2
    ])

def get_val_transforms(img_size: int = 224) -> A.Compose:
    """
    Get validation/test image transformations (no augmentation).

    Args:
        img_size: Target image size
    
    Returns:
        Albumentations composition
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

def get_test_time_augmentation(img_size: int = 224, n_aug: int = 5) -> list:
    """
    Get test-time augmentation transforms for ensemble predictions.

    Args:
        img_size: Target image size
        n_aug: Number of augmentation variants
    
    Returns:
        List of augmentation compositions
    """
    base_transform = [
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ]

    augmentation = [
        A.Compose(base_transform),
        A.Compose([A.HorizontalFlip(p=1.0)] + base_transform),
        A.Compose([A.Rotate(limit=10, p=1.0)] + base_transform),
        A.Compose([A.RandomBrightnessContrast(p=1.0)] + base_transform),
        A.Compose([A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=1.0)] + base_transform),

    ]

    return augmentation[:n_aug]