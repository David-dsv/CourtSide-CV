"""
Training Script for Action Classifier
Train X3D or SlowFast model on tennis action dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from loguru import logger
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.action_classifier import TennisActionClassifier
from training.dataset import TennisActionDataset
from training.transforms import get_train_transforms, get_val_transforms
from utils.file_utils import load_yaml, save_json, ensure_dir


class Trainer:
    """Training manager for action classifier"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        device: str,
        output_dir: str,
        num_epochs: int,
        grad_clip: float = 1.0,
        save_interval: int = 5,
        val_interval: int = 1
    ):
        """
        Initialize trainer

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            output_dir: Output directory for checkpoints
            num_epochs: Number of epochs
            grad_clip: Gradient clipping value
            save_interval: Save checkpoint every N epochs
            val_interval: Validate every N epochs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.grad_clip = grad_clip
        self.save_interval = save_interval
        self.val_interval = val_interval

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()

        self.best_val_acc = 0.0
        self.train_history = []
        self.val_history = []

    def train_epoch(self, epoch: int) -> dict:
        """Train one epoch"""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}")

        for batch_idx, (clips, labels) in enumerate(pbar):
            clips = clips.to(self.device)
            labels = labels.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(clips)
            loss = self.criterion(outputs, labels)

            # Backward
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100.0 * correct / total
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    @torch.no_grad()
    def validate(self) -> dict:
        """Validate model"""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for clips, labels in tqdm(self.val_loader, desc="Validating"):
            clips = clips.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(clips)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_history': self.train_history,
            'val_history': self.val_history
        }

        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")

        # Save latest
        latest_path = self.output_dir / "latest_model.pth"
        torch.save(checkpoint, latest_path)

    def train(self):
        """Main training loop"""
        logger.info(f"Starting training for {self.num_epochs} epochs")

        for epoch in range(1, self.num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_history.append(train_metrics)

            logger.info(
                f"Epoch {epoch}/{self.num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%"
            )

            # Validate
            if epoch % self.val_interval == 0:
                val_metrics = self.validate()
                self.val_history.append(val_metrics)

                logger.info(
                    f"Epoch {epoch}/{self.num_epochs} - "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.2f}%"
                )

                # Check if best
                is_best = val_metrics['accuracy'] > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_metrics['accuracy']
                    logger.info(f"New best validation accuracy: {self.best_val_acc:.2f}%")

                # Save checkpoint
                if epoch % self.save_interval == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)

            # Step scheduler
            self.scheduler.step()

        # Save final training history
        history_path = self.output_dir / "training_history.json"
        save_json({
            'train': self.train_history,
            'val': self.val_history,
            'best_val_acc': self.best_val_acc
        }, str(history_path))

        logger.info(f"Training completed. Best val accuracy: {self.best_val_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Train Tennis Action Classifier")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--output-dir", type=str, default="runs/train", help="Output directory")
    parser.add_argument("--model-type", type=str, default="x3d", choices=["x3d", "slowfast"])
    parser.add_argument("--model-size", type=str, default="m", choices=["s", "m", "l"])
    parser.add_argument("--num-classes", type=int, default=6, help="Number of classes")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="Weight decay")
    parser.add_argument("--clip-length", type=int, default=32, help="Clip length")
    parser.add_argument("--spatial-size", type=int, default=224, help="Spatial size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Setup logging
    logger.add(
        Path(args.output_dir) / "training.log",
        rotation="10 MB",
        level="INFO"
    )

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create datasets
    train_transform = get_train_transforms(
        spatial_size=args.spatial_size,
        clip_length=args.clip_length,
        augment=True
    )

    val_transform = get_val_transforms(
        spatial_size=args.spatial_size,
        clip_length=args.clip_length
    )

    train_dataset = TennisActionDataset(
        data_root=args.data_root,
        split="train",
        clip_length=args.clip_length,
        spatial_size=args.spatial_size,
        transform=train_transform
    )

    val_dataset = TennisActionDataset(
        data_root=args.data_root,
        split="val",
        clip_length=args.clip_length,
        spatial_size=args.spatial_size,
        transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model
    model = TennisActionClassifier(
        model_type=args.model_type,
        model_size=args.model_size,
        num_classes=args.num_classes,
        pretrained=True
    )
    model = model.to(device)

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        grad_clip=1.0,
        save_interval=5,
        val_interval=1
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
