"""
Custom metrics for model evaluation.
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Tuple


class MetricsCalculator:
    """Calculate various evaluation metrics."""
    
    @staticmethod
    def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate accuracy."""
        correct = (predictions == targets).sum().item()
        total = targets.size(0)
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
        """Calculate top-k accuracy."""
        _, top_k_preds = logits.topk(k, dim=1)
        correct = top_k_preds.eq(targets.view(-1, 1).expand_as(top_k_preds)).any(dim=1)
        return correct.float().mean().item()
    
    @staticmethod
    def f1_score_multiclass(predictions: np.ndarray, targets: np.ndarray, average: str = 'weighted') -> float:
        """Calculate F1 score for multiclass."""
        return f1_score(targets, predictions, average=average, zero_division=0)
    
    @staticmethod
    def precision_recall(predictions: np.ndarray, targets: np.ndarray, average: str = 'weighted') -> Tuple[float, float]:
        """Calculate precision and recall."""
        precision = precision_score(targets, predictions, average=average, zero_division=0)
        recall = recall_score(targets, predictions, average=average, zero_division=0)
        return precision, recall
    
    @staticmethod
    def mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate Mean Absolute Error."""
        return torch.abs(predictions - targets).mean().item()
    
    @staticmethod
    def mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate Mean Squared Error."""
        return torch.pow(predictions - targets, 2).mean().item()
    
    @staticmethod
    def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate Root Mean Squared Error."""
        return torch.sqrt(torch.pow(predictions - targets, 2).mean()).item()


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """Update statistics."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
    
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"