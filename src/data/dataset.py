"""
PyTorch Dataset classes for multimodal product review data.
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
from transformers import AutoTokenizer


class MultimodalProductDataset(Dataset):
    """
    Dataset for multimodal product reviews (image + text).
    
    Args:
        data_df: DataFrame with columns ['image_path', 'review_text', 'category', 
                 'sentiment', 'recommendation_score']
        transform: Image transformations
        tokenizer: Text tokenizer
        max_length: Maximum sequence length for text
    """
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        transform: Optional[object] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 128,
        category_to_idx: Optional[Dict[str, int]] = None,
        sentiment_to_idx: Optional[Dict[str, int]] = None
    ):
        self.data = data_df.reset_index(drop=True)
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label mappings
        if category_to_idx is None:
            unique_categories = sorted(self.data['category'].unique())
            self.category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
        else:
            self.category_to_idx = category_to_idx
        
        if sentiment_to_idx is None:
            self.sentiment_to_idx = {'negative': 0, 'neutral': 1, 'positive': 2}
        else:
            self.sentiment_to_idx = sentiment_to_idx
        
        # Reverse mappings for inference
        self.idx_to_category = {v: k for k, v in self.category_to_idx.items()}
        self.idx_to_sentiment = {v: k for k, v in self.sentiment_to_idx.items()}
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
                - image: Tensor of shape (3, H, W)
                - input_ids: Tensor of shape (max_length,)
                - attention_mask: Tensor of shape (max_length,)
                - category_label: Tensor (scalar)
                - sentiment_label: Tensor (scalar)
                - recommendation_score: Tensor (scalar, 0-1)
        """
        row = self.data.iloc[idx]
        
        # Load and transform image
        image_path = Path(row['image_path'])
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image=np.array(image))['image']
            else:
                # Default: convert to tensor
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        except Exception as e:
            # Return a black image if loading fails
            print(f"Error loading image {image_path}: {e}")
            image = torch.zeros(3, 224, 224)
        
        # Tokenize text
        review_text = str(row['review_text'])
        if self.tokenizer:
            encoding = self.tokenizer(
                review_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
        else:
            # Dummy tokens
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        
        # Labels
        category_label = self.category_to_idx.get(row['category'], 0)
        sentiment_label = self.sentiment_to_idx.get(row['sentiment'], 1)
        recommendation_score = float(row.get('recommendation_score', 0.5))
        
        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'category_label': torch.tensor(category_label, dtype=torch.long),
            'sentiment_label': torch.tensor(sentiment_label, dtype=torch.long),
            'recommendation_score': torch.tensor(recommendation_score, dtype=torch.float)
        }


class VisionOnlyDataset(Dataset):
    """Dataset for vision-only training."""
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        transform: Optional[object] = None,
        category_to_idx: Optional[Dict[str, int]] = None
    ):
        self.data = data_df.reset_index(drop=True)
        self.transform = transform
        
        if category_to_idx is None:
            unique_categories = sorted(self.data['category'].unique())
            self.category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
        else:
            self.category_to_idx = category_to_idx
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]
        
        # Load image
        image_path = Path(row['image_path'])
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image=np.array(image))['image']
            else:
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = torch.zeros(3, 224, 224)
        
        # Label
        category_label = self.category_to_idx.get(row['category'], 0)
        
        return image, torch.tensor(category_label, dtype=torch.long)


class NLPOnlyDataset(Dataset):
    """Dataset for NLP-only training."""
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        sentiment_to_idx: Optional[Dict[str, int]] = None
    ):
        self.data = data_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if sentiment_to_idx is None:
            self.sentiment_to_idx = {'negative': 0, 'neutral': 1, 'positive': 2}
        else:
            self.sentiment_to_idx = sentiment_to_idx
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        
        # Tokenize
        review_text = str(row['review_text'])
        encoding = self.tokenizer(
            review_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Label
        sentiment_label = self.sentiment_to_idx.get(row['sentiment'], 1)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(sentiment_label, dtype=torch.long)
        }


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 4,
    dataset_type: str = "multimodal"
) -> Tuple:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        train_df, val_df, test_df: DataFrames with product data
        batch_size: Batch size
        num_workers: Number of worker processes
        dataset_type: "multimodal", "vision", or "nlp"
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    from src.data.augmentation import get_train_transforms, get_val_transforms
    from transformers import AutoTokenizer
    from src.config import config
    
    if dataset_type == "multimodal":
        train_transform = get_train_transforms()
        val_transform = get_val_transforms()
        tokenizer = AutoTokenizer.from_pretrained(config.nlp.model_name)
        
        train_dataset = MultimodalProductDataset(
            train_df, transform=train_transform, tokenizer=tokenizer
        )
        val_dataset = MultimodalProductDataset(
            val_df, transform=val_transform, tokenizer=tokenizer,
            category_to_idx=train_dataset.category_to_idx,
            sentiment_to_idx=train_dataset.sentiment_to_idx
        )
        test_dataset = MultimodalProductDataset(
            test_df, transform=val_transform, tokenizer=tokenizer,
            category_to_idx=train_dataset.category_to_idx,
            sentiment_to_idx=train_dataset.sentiment_to_idx
        )
    
    elif dataset_type == "vision":
        train_transform = get_train_transforms()
        val_transform = get_val_transforms()
        
        train_dataset = VisionOnlyDataset(train_df, transform=train_transform)
        val_dataset = VisionOnlyDataset(
            val_df, transform=val_transform,
            category_to_idx=train_dataset.category_to_idx
        )
        test_dataset = VisionOnlyDataset(
            test_df, transform=val_transform,
            category_to_idx=train_dataset.category_to_idx
        )
    
    elif dataset_type == "nlp":
        tokenizer = AutoTokenizer.from_pretrained(config.nlp.model_name)
        
        train_dataset = NLPOnlyDataset(train_df, tokenizer=tokenizer)
        val_dataset = NLPOnlyDataset(
            val_df, tokenizer=tokenizer,
            sentiment_to_idx=train_dataset.sentiment_to_idx
        )
        test_dataset = NLPOnlyDataset(
            test_df, tokenizer=tokenizer,
            sentiment_to_idx=train_dataset.sentiment_to_idx
        )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset