"""
Training utilities for vision, NLP, and fusion models.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import wandb
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer:
    """
    Base trainer class with common training functionality.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object] = None,
        device: str = "cuda",
        num_epochs: int = 10,
        save_dir: Path = Path("models"),
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        gradient_clip: Optional[float] = 1.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_clip = gradient_clip
        
        # Logging
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.save_dir / "tensorboard")
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._batch_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.forward_pass(batch)
            loss = self.compute_loss(outputs, batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            
            # Calculate accuracy
            acc = self.compute_accuracy(outputs, batch)
            
            # Update metrics
            total_loss += loss.item()
            correct += acc * batch[self._get_batch_size_key(batch)]
            total += batch[self._get_batch_size_key(batch)]
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                batch = self._batch_to_device(batch)
                
                outputs = self.forward_pass(batch)
                loss = self.compute_loss(outputs, batch)
                acc = self.compute_accuracy(outputs, batch)
                
                total_loss += loss.item()
                correct += acc * batch[self._get_batch_size_key(batch)]
                total += batch[self._get_batch_size_key(batch)]
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Logging
            self._log_metrics(train_loss, train_acc, val_loss, val_acc)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_loss.pth")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint("best_acc.pth")
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pth")
            
            print(f"Epoch {epoch + 1}/{self.num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        print("Training completed!")
        self.save_checkpoint("final.pth")
        
        if self.use_tensorboard:
            self.writer.close()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.save_dir / filename)
        print(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accs = checkpoint['train_accs']
        self.val_accs = checkpoint['val_accs']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        
        print(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def _batch_to_device(self, batch):
        """Move batch to device."""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for v in batch]
        else:
            return batch.to(self.device)
    
    def _get_batch_size_key(self, batch):
        """Get batch size from batch."""
        if isinstance(batch, dict):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    return batch[key].size(0)
        return 1
    
    def _log_metrics(self, train_loss, train_acc, val_loss, val_acc):
        """Log metrics to wandb and tensorboard."""
        if self.use_wandb:
            wandb.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'epoch': self.current_epoch,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
        
        if self.use_tensorboard:
            self.writer.add_scalar('Loss/train', train_loss, self.current_epoch)
            self.writer.add_scalar('Loss/val', val_loss, self.current_epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, self.current_epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, self.current_epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], self.current_epoch)
    
    def forward_pass(self, batch):
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError
    
    def compute_loss(self, outputs, batch):
        """Compute loss - to be implemented by subclasses."""
        raise NotImplementedError
    
    def compute_accuracy(self, outputs, batch):
        """Compute accuracy - to be implemented by subclasses."""
        raise NotImplementedError


class VisionTrainer(BaseTrainer):
    """Trainer for vision models."""
    
    def forward_pass(self, batch):
        if isinstance(batch, dict):
            return self.model(batch['image'])
        else:
            images, _ = batch
            return self.model(images)
    
    def compute_loss(self, outputs, batch):
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        labels = batch['category_label'] if isinstance(batch, dict) else batch[1]
        return self.criterion(logits, labels)
    
    def compute_accuracy(self, outputs, batch):
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        labels = batch['category_label'] if isinstance(batch, dict) else batch[1]
        preds = torch.argmax(logits, dim=1)
        return (preds == labels).float().mean().item()


class NLPTrainer(BaseTrainer):
    """Trainer for NLP models."""
    
    def forward_pass(self, batch):
        return self.model(batch['input_ids'], batch['attention_mask'])
    
    def compute_loss(self, outputs, batch):
        logits = outputs['logits']
        labels = batch['labels'] if 'labels' in batch else batch['sentiment_label']
        return self.criterion(logits, labels)
    
    def compute_accuracy(self, outputs, batch):
        logits = outputs['logits']
        labels = batch['labels'] if 'labels' in batch else batch['sentiment_label']
        preds = torch.argmax(logits, dim=1)
        return (preds == labels).float().mean().item()


class FusionTrainer(BaseTrainer):
    """Trainer for fusion models with multi-task learning."""
    
    def __init__(self, *args, category_weight=1.0, sentiment_weight=1.0, 
                 recommendation_weight=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.category_weight = category_weight
        self.sentiment_weight = sentiment_weight
        self.recommendation_weight = recommendation_weight
        
        # Multiple criterions
        self.category_criterion = nn.CrossEntropyLoss()
        self.sentiment_criterion = nn.CrossEntropyLoss()
        self.recommendation_criterion = nn.MSELoss()
    
    def forward_pass(self, batch):
        return self.model(
            batch['image'],
            batch['input_ids'],
            batch['attention_mask']
        )
    
    def compute_loss(self, outputs, batch):
        # Multi-task loss
        category_loss = self.category_criterion(
            outputs['category_logits'],
            batch['category_label']
        )
        
        sentiment_loss = self.sentiment_criterion(
            outputs['sentiment_logits'],
            batch['sentiment_label']
        )
        
        recommendation_loss = self.recommendation_criterion(
            outputs['recommendation_score'].squeeze(),
            batch['recommendation_score']
        )
        
        total_loss = (
            self.category_weight * category_loss +
            self.sentiment_weight * sentiment_loss +
            self.recommendation_weight * recommendation_loss
        )
        
        return total_loss
    
    def compute_accuracy(self, outputs, batch):
        # Category accuracy
        cat_preds = torch.argmax(outputs['category_logits'], dim=1)
        cat_acc = (cat_preds == batch['category_label']).float().mean().item()
        
        # Sentiment accuracy
        sent_preds = torch.argmax(outputs['sentiment_logits'], dim=1)
        sent_acc = (sent_preds == batch['sentiment_label']).float().mean().item()
        
        # Average accuracy
        return (cat_acc + sent_acc) / 2