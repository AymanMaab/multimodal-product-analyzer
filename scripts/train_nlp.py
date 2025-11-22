"""
Training script for NLP model.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import argparse

from src.config import config
from src.data.dataset import create_dataloaders
from src.models.nlp_model import SentimentClassifier
from src.training.trainer import NLPTrainer
from src.data.preprocessor import split_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train NLP Model")
    parser.add_argument('--data_csv', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='bert', choices=['bert', 'roberta'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_wandb', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("NLP MODEL TRAINING")
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
        dataset_type="nlp"
    )
    
    # Initialize model
    print("\n[3/5] Initializing model...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    model = SentimentClassifier(
        model_name=config.nlp.model_name,
        num_classes=len(train_dataset.sentiment_to_idx)
    )
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    print("\n[4/5] Setting up optimizer...")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=config.nlp.weight_decay)
    
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.nlp.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print("\n[5/5] Starting training...")
    trainer = NLPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=str(device),
        num_epochs=args.epochs,
        save_dir=config.paths.MODELS_DIR / "nlp",
        use_wandb=args.use_wandb
    )
    
    trainer.train()
    
    # Save mappings
    torch.save(
        {'sentiment_to_idx': train_dataset.sentiment_to_idx},
        config.paths.MODELS_DIR / "nlp" / "mappings.pth"
    )
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()