"""
Training script for fusion model.
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
from src.models.nlp_model import SentimentClassifier
from src.models.fusion_model import MultimodalFusionModel
from src.training.trainer import FusionTrainer
from src.data.preprocessor import split_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Fusion Model")
    parser.add_argument('--data_csv', type=str, required=True,
                       help='Path to processed dataset CSV')
    parser.add_argument('--vision_checkpoint', type=str, default=None,
                       help='Path to pretrained vision model')
    parser.add_argument('--nlp_checkpoint', type=str, default=None,
                       help='Path to pretrained NLP model')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--freeze_encoders', action='store_true',
                       help='Freeze pretrained encoder weights')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 80)
    print("MULTIMODAL FUSION MODEL TRAINING")
    print("=" * 80)
    
    # Load and split dataset
    print("\n[1/6] Loading dataset...")
    df = pd.read_csv(args.data_csv)
    print(f"Total samples: {len(df)}")
    
    train_df, val_df, test_df = split_dataset(
        df,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        random_state=config.data.random_seed
    )
    
    # Create dataloaders
    print("\n[2/6] Creating dataloaders...")
    train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
        train_df, val_df, test_df,
        batch_size=args.batch_size,
        num_workers=4,
        dataset_type="multimodal"
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Initialize models
    print("\n[3/6] Initializing models...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Vision encoder
    vision_encoder = VisionEncoder(
        model_name=config.vision.model_name,
        num_classes=len(train_dataset.category_to_idx)
    )
    
    if args.vision_checkpoint:
        print(f"Loading vision checkpoint: {args.vision_checkpoint}")
        checkpoint = torch.load(args.vision_checkpoint, map_location=device)
        vision_encoder.load_state_dict(checkpoint['model_state_dict'])
    
    # NLP encoder
    nlp_encoder = SentimentClassifier(
        model_name=config.nlp.model_name,
        num_classes=len(train_dataset.sentiment_to_idx)
    )
    
    if args.nlp_checkpoint:
        print(f"Loading NLP checkpoint: {args.nlp_checkpoint}")
        checkpoint = torch.load(args.nlp_checkpoint, map_location=device)
        nlp_encoder.load_state_dict(checkpoint['model_state_dict'])
    
    # Fusion model
    fusion_model = MultimodalFusionModel(
        vision_encoder=vision_encoder,
        nlp_encoder=nlp_encoder,
        num_categories=len(train_dataset.category_to_idx),
        num_sentiments=len(train_dataset.sentiment_to_idx),
        fusion_hidden_dims=config.fusion.fusion_hidden_dims,
        dropout=config.fusion.dropout_rate,
        freeze_encoders=args.freeze_encoders
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in fusion_model.parameters())
    trainable_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer and scheduler
    print("\n[4/6] Setting up optimizer...")
    optimizer = AdamW(
        fusion_model.parameters(),
        lr=args.lr,
        weight_decay=config.fusion.weight_decay
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Criterion (will be handled in trainer)
    criterion = None  # Multi-task loss in FusionTrainer
    
    # Initialize trainer
    print("\n[5/6] Initializing trainer...")
    trainer = FusionTrainer(
        model=fusion_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=str(device),
        num_epochs=args.epochs,
        save_dir=config.paths.MODELS_DIR / "fusion",
        use_wandb=args.use_wandb,
        use_tensorboard=True,
        category_weight=config.fusion.category_weight,
        sentiment_weight=config.fusion.sentiment_weight,
        recommendation_weight=config.fusion.recommendation_weight
    )
    
    # Train model
    print("\n[6/6] Starting training...")
    print("=" * 80)
    
    if args.use_wandb:
        import wandb
        wandb.init(
            project="multimodal-product-analyzer",
            name="fusion-model-training",
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "freeze_encoders": args.freeze_encoders,
                "architecture": "MultimodalFusion",
                "vision_model": config.vision.model_name,
                "nlp_model": config.nlp.model_name
            }
        )
    
    trainer.train()
    
    # Save label mappings
    label_mappings = {
        'category_to_idx': train_dataset.category_to_idx,
        'sentiment_to_idx': train_dataset.sentiment_to_idx,
        'idx_to_category': train_dataset.idx_to_category,
        'idx_to_sentiment': train_dataset.idx_to_sentiment
    }
    
    torch.save(label_mappings, config.paths.MODELS_DIR / "fusion" / "label_mappings.pth")
    print(f"\nLabel mappings saved to {config.paths.MODELS_DIR / 'fusion' / 'label_mappings.pth'}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
    print(f"Models saved to: {config.paths.MODELS_DIR / 'fusion'}")
    print("=" * 80)


if __name__ == "__main__":
    main()