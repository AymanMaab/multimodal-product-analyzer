"""
Model evaluation script.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pandas as pd
import argparse

from src.config import config
from src.data.dataset import create_dataloaders
from src.models.fusion_model import MultimodalFusionModel
from src.models.vision_model import VisionEncoder
from src.models.nlp_model import SentimentClassifier
from src.training.evaluator import ModelEvaluator
from src.data.preprocessor import split_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Models")
    parser.add_argument('--data_csv', type=str, required=True)
    parser.add_argument('--fusion_checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='results')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    # Load data
    print("\n[1/4] Loading dataset...")
    df = pd.read_csv(args.data_csv)
    _, _, test_df = split_dataset(df)
    
    # Create dataloader
    print("\n[2/4] Creating dataloader...")
    _, _, test_loader, train_dataset = create_dataloaders(
        test_df, test_df, test_df,  # Using test_df for all
        batch_size=args.batch_size,
        dataset_type="multimodal"
    )
    
    # Load model
    print("\n[3/4] Loading model...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Initialize encoders
    vision_encoder = VisionEncoder(
        num_classes=len(train_dataset.category_to_idx)
    )
    nlp_encoder = SentimentClassifier(
        num_classes=len(train_dataset.sentiment_to_idx)
    )
    
    # Initialize fusion model
    fusion_model = MultimodalFusionModel(
        vision_encoder=vision_encoder,
        nlp_encoder=nlp_encoder,
        num_categories=len(train_dataset.category_to_idx),
        num_sentiments=len(train_dataset.sentiment_to_idx)
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.fusion_checkpoint, map_location=device)
    fusion_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    print("\n[4/4] Evaluating model...")
    evaluator = ModelEvaluator(
        model=fusion_model,
        test_loader=test_loader,
        device=str(device),
        save_dir=Path(args.output)
    )
    
    results = evaluator.evaluate_fusion_model(
        category_names=list(train_dataset.idx_to_category.values()),
        sentiment_names=list(train_dataset.idx_to_sentiment.values())
    )
    
    print("\nEvaluation completed! Results saved to:", args.output)


if __name__ == "__main__":
    main()