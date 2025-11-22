"""
Training script for vision model.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import argparse

from src.config import config
from src.data.dataset import create_dataloaders
from src.models.vision_model import VisionEncoder
from src.training.trainer import VisionTrainer
from src.data.preprocessor import split_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train Vision Model")
    parser.add_argument('--data_csv', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='vit', choices=['vit', 'cnn'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_wandb', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("VISION MODEL TRAINING")
    print("=" * 80)
    
    # Load data
    print("\n[1/5] Loading dataset...")
    df = pd.read_csv(args.data_csv)
    train_df, val_df, test_df = split_dataset(df)
    
    # Create dataloaders
    print("\n[2/5] Creating dataloaders...")
    train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
        train_df, val_df, test_df,
        batch_size=args.batch_size,
        dataset_type="vision"
    )
    
    # Initialize model
    print("\n[3/5] Initializing model...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    model = VisionEncoder(
        model_name=config.vision.model_name,
        num_classes=len(train_dataset.category_to_idx)
    )
    
    print(f"Model: {args.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    print("\n[4/5] Setting up optimizer...")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=config.vision.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print("\n[5/5] Starting training...")
    trainer = VisionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=str(device),
        num_epochs=args.epochs,
        save_dir=config.paths.MODELS_DIR / "vision",
        use_wandb=args.use_wandb
    )
    
    trainer.train()
    
    # Save mappings
    torch.save(
        {'category_to_idx': train_dataset.category_to_idx},
        config.paths.MODELS_DIR / "vision" / "mappings.pth"
    )
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()