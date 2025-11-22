#!/bin/bash
# Quickstart script for Multimodal Product Review Analyzer

set -e

echo "=========================================="
echo "Multimodal Product Analyzer - Quickstart"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo -e "${RED}Error: Python 3.8+ required. Found: $python_version${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $python_version${NC}"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install dependencies
echo ""
echo "Installing dependencies (this may take a while)..."
pip install -r requirements.txt > /dev/null 2>&1
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Download NLTK data
echo ""
echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"
echo -e "${GREEN}✓ NLTK data downloaded${NC}"

# Create directories
echo ""
echo "Creating project directories..."
mkdir -p data/raw data/processed models/vision models/nlp models/fusion logs results/figures
echo -e "${GREEN}✓ Directories created${NC}"

# Create .env file
echo ""
echo "Creating .env file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
else
    echo -e "${YELLOW}.env file already exists${NC}"
fi

# Generate sample dataset
echo ""
echo "Generating sample dataset..."
python3 scripts/download_data.py --create_sample --output data/raw
echo -e "${GREEN}✓ Sample dataset created${NC}"

# Preprocess data
echo ""
echo "Preprocessing data..."
python3 scripts/preprocess_data.py \
    --images_dir data/raw/images \
    --reviews_csv data/raw/reviews.csv \
    --output data/processed/dataset.csv \
    --sample_size 100
echo -e "${GREEN}✓ Data preprocessed${NC}"

# Display next steps
echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Train models:"
echo "   ${YELLOW}python scripts/train_vision.py --data_csv data/processed/dataset.csv --epochs 5${NC}"
echo "   ${YELLOW}python scripts/train_nlp.py --data_csv data/processed/dataset.csv --epochs 3${NC}"
echo "   ${YELLOW}python scripts/train_fusion.py --data_csv data/processed/dataset.csv --epochs 5${NC}"
echo ""
echo "2. Start API server:"
echo "   ${YELLOW}uvicorn src.api.main:app --reload${NC}"
echo ""
echo "3. Launch dashboard (in new terminal):"
echo "   ${YELLOW}streamlit run app/streamlit_app.py${NC}"
echo ""
echo "4. Or use Docker:"
echo "   ${YELLOW}cd docker && docker-compose up${NC}"
echo ""
echo "=========================================="
echo "For more information, see README.md"
echo "=========================================="