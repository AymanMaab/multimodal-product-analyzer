"""
Data preprocessing script.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
from src.data.preprocessor import prepare_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument('--images_dir', type=str, required=True,
                       help='Directory containing product images')
    parser.add_argument('--reviews_csv', type=str, required=True,
                       help='Path to reviews CSV file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV path')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Sample size (optional)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)
    
    df = prepare_dataset(
        images_dir=Path(args.images_dir),
        reviews_csv=Path(args.reviews_csv),
        output_csv=Path(args.output),
        sample_size=args.sample_size
    )
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()