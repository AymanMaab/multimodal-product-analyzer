"""
Model evaluation utilities with comprehensive metrics.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score,
    mean_absolute_error, mean_squared_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm


class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    """
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = "cuda",
        save_dir: Path = Path("results")
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
    
    def evaluate_classification(
        self,
        task_name: str,
        label_key: str,
        logits_key: str,
        class_names: List[str]
    ) -> Dict:
        """
        Evaluate classification task.
        
        Args:
            task_name: Name of the task (e.g., 'category', 'sentiment')
            label_key: Key for true labels in batch
            logits_key: Key for logits in model output
            class_names: List of class names
        
        Returns:
            Dictionary with evaluation metrics
        """
        all_preds = []
        all_labels = []
        all_probs = []
        
        print(f"Evaluating {task_name}...")
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                batch = self._batch_to_device(batch)
                
                # Forward pass
                outputs = self._forward_pass(batch)
                logits = outputs[logits_key]
                labels = batch[label_key]
                
                # Get predictions
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, support = \
            precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Classification report
        report = classification_report(
            all_labels, all_preds,
            target_names=class_names,
            zero_division=0
        )
        
        # ROC AUC (if binary or multi-class)
        try:
            if len(class_names) == 2:
                auc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        except:
            auc = None
        
        # Plot confusion matrix
        self._plot_confusion_matrix(
            cm, class_names, 
            title=f"{task_name.title()} Confusion Matrix",
            save_path=self.save_dir / f"{task_name}_confusion_matrix.png"
        )
        
        # Results dictionary
        results = {
            'task': task_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc) if auc is not None else None,
            'per_class_metrics': {
                class_names[i]: {
                    'precision': float(per_class_precision[i]),
                    'recall': float(per_class_recall[i]),
                    'f1_score': float(per_class_f1[i]),
                    'support': int(support[i])
                }
                for i in range(len(class_names))
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        return results
    
    def evaluate_regression(
        self,
        task_name: str,
        label_key: str,
        pred_key: str
    ) -> Dict:
        """
        Evaluate regression task (e.g., recommendation score).
        
        Args:
            task_name: Name of the task
            label_key: Key for true values
            pred_key: Key for predictions
        
        Returns:
            Dictionary with regression metrics
        """
        all_preds = []
        all_labels = []
        
        print(f"Evaluating {task_name}...")
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                batch = self._batch_to_device(batch)
                
                outputs = self._forward_pass(batch)
                preds = outputs[pred_key].squeeze()
                labels = batch[label_key]
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        mae = mean_absolute_error(all_labels, all_preds)
        mse = mean_squared_error(all_labels, all_preds)
        rmse = np.sqrt(mse)
        
        # R² score
        ss_res = np.sum((all_labels - all_preds) ** 2)
        ss_tot = np.sum((all_labels - np.mean(all_labels)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Plot predictions vs actual
        self._plot_regression(
            all_labels, all_preds,
            title=f"{task_name.title()} Predictions vs Actual",
            save_path=self.save_dir / f"{task_name}_predictions.png"
        )
        
        results = {
            'task': task_name,
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2_score': float(r2)
        }
        
        return results
    
    def evaluate_fusion_model(
        self,
        category_names: List[str],
        sentiment_names: List[str]
    ) -> Dict:
        """
        Evaluate complete fusion model with all tasks.
        
        Returns:
            Complete evaluation results
        """
        results = {}
        
        # Evaluate category classification
        results['category'] = self.evaluate_classification(
            task_name='category',
            label_key='category_label',
            logits_key='category_logits',
            class_names=category_names
        )
        
        # Evaluate sentiment classification
        results['sentiment'] = self.evaluate_classification(
            task_name='sentiment',
            label_key='sentiment_label',
            logits_key='sentiment_logits',
            class_names=sentiment_names
        )
        
        # Evaluate recommendation score
        results['recommendation'] = self.evaluate_regression(
            task_name='recommendation',
            label_key='recommendation_score',
            pred_key='recommendation_score'
        )
        
        # Calculate overall performance
        results['overall'] = {
            'avg_accuracy': (results['category']['accuracy'] + 
                           results['sentiment']['accuracy']) / 2,
            'avg_f1': (results['category']['f1_score'] + 
                      results['sentiment']['f1_score']) / 2,
            'recommendation_mae': results['recommendation']['mae']
        }
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _forward_pass(self, batch):
        """Forward pass through model."""
        if hasattr(self.model, 'forward'):
            if 'image' in batch and 'input_ids' in batch:
                # Fusion model
                return self.model(
                    batch['image'],
                    batch['input_ids'],
                    batch['attention_mask']
                )
            elif 'image' in batch:
                # Vision model
                return self.model(batch['image'])
            else:
                # NLP model
                return self.model(batch['input_ids'], batch['attention_mask'])
    
    def _batch_to_device(self, batch):
        """Move batch to device."""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        return batch
    
    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        title: str,
        save_path: Path
    ):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix to {save_path}")
    
    def _plot_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str,
        save_path: Path
    ):
        """Plot regression predictions vs actual."""
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], 
                [y_true.min(), y_true.max()], 
                'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved regression plot to {save_path}")
    
    def _save_results(self, results: Dict):
        """Save results to JSON."""
        output_path = self.save_dir / "evaluation_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nEvaluation results saved to {output_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"\nCategory Classification:")
        print(f"  Accuracy: {results['category']['accuracy']:.4f}")
        print(f"  F1-Score: {results['category']['f1_score']:.4f}")
        print(f"\nSentiment Analysis:")
        print(f"  Accuracy: {results['sentiment']['accuracy']:.4f}")
        print(f"  F1-Score: {results['sentiment']['f1_score']:.4f}")
        print(f"\nRecommendation Score:")
        print(f"  MAE: {results['recommendation']['mae']:.4f}")
        print(f"  RMSE: {results['recommendation']['rmse']:.4f}")
        print(f"  R²: {results['recommendation']['r2_score']:.4f}")
        print(f"\nOverall Performance:")
        print(f"  Avg Accuracy: {results['overall']['avg_accuracy']:.4f}")
        print(f"  Avg F1-Score: {results['overall']['avg_f1']:.4f}")
        print("=" * 80)