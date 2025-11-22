# Makefile for Multimodal Product Review Analyzer

.PHONY: help install clean test lint format train api dashboard docker

help:
	@echo "Multimodal Product Review Analyzer - Available Commands:"
	@echo ""
	@echo "  make install       - Install dependencies"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"
	@echo "  make data          - Generate sample dataset"
	@echo "  make train-vision  - Train vision model"
	@echo "  make train-nlp     - Train NLP model"
	@echo "  make train-fusion  - Train fusion model"
	@echo "  make train-all     - Train all models"
	@echo "  make evaluate      - Evaluate models"
	@echo "  make api           - Start API server"
	@echo "  make dashboard     - Start Streamlit dashboard"
	@echo "  make docker-build  - Build Docker images"
	@echo "  make docker-up     - Start Docker services"
	@echo "  make docker-down   - Stop Docker services"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf build/ dist/ *.egg-info htmlcov/ .pytest_cache/ .coverage

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ scripts/ tests/ --max-line-length=100
	mypy src/ --ignore-missing-imports

format:
	black src/ scripts/ tests/ app/
	isort src/ scripts/ tests/ app/

data:
	python scripts/download_data.py --create_sample --output data/raw
	python scripts/preprocess_data.py \
		--images_dir data/raw/images \
		--reviews_csv data/raw/reviews.csv \
		--output data/processed/dataset.csv

train-vision:
	python scripts/train_vision.py \
		--data_csv data/processed/dataset.csv \
		--epochs 30 \
		--batch_size 32 \
		--device cuda

train-nlp:
	python scripts/train_nlp.py \
		--data_csv data/processed/dataset.csv \
		--epochs 10 \
		--batch_size 16 \
		--device cuda

train-fusion:
	python scripts/train_fusion.py \
		--data_csv data/processed/dataset.csv \
		--vision_checkpoint models/vision/best_acc.pth \
		--nlp_checkpoint models/nlp/best_acc.pth \
		--epochs 20 \
		--batch_size 32 \
		--device cuda

train-all: train-vision train-nlp train-fusion

evaluate:
	python scripts/evaluate.py \
		--data_csv data/processed/dataset.csv \
		--fusion_checkpoint models/fusion/best_acc.pth \
		--output results

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run app/streamlit_app.py

docker-build:
	cd docker && docker-compose build

docker-up:
	cd docker && docker-compose up -d

docker-down:
	cd docker && docker-compose down

docker-logs:
	cd docker && docker-compose logs -f

setup:
	chmod +x quickstart.sh
	./quickstart.sh