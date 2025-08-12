

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import numpy as np
import argparse
import os
from pathlib import Path
from typing import Dict, Any, Optional
import random

from ..models.enhanced_unetmamba import EnhancedUNetMamba
from ..models.losses import EnhancedMultiScaleLoss
from ..utils.metrics import EnhancedEvaluator
from ..utils.config import load_config


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EnhancedSupervisionTrain(pl.LightningModule):
    """Enhanced PyTorch Lightning module for training."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        
        # Initialize model
        self.net = EnhancedUNetMamba(
            num_classes=config['num_classes'],
            channels=config.get('channels', [64, 128, 256, 512]),
            use_local_supervision=config.get('use_local_supervision', True)
        )
        
        # Initialize enhanced loss
        self.loss_fn = EnhancedMultiScaleLoss(
            num_classes=config['num_classes'],
            **config.get('loss_config', {})
        )
        
        # Initialize metrics
        self.metrics_train = EnhancedEvaluator(num_class=config['num_classes'])
        self.metrics_val = EnhancedEvaluator(num_class=config['num_classes'])
        
        # Learning rate for logging
        self.learning_rate = config.get('learning_rate', 6e-4)
        
        # Validation metrics for early stopping
        self.best_val_miou = 0.0
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        return self.net(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step."""
        img = batch['img']
        mask = batch['gt_semantic_seg']
        
        # Forward pass
        outputs = self.forward(img)
        
        # Compute losses
        loss_dict = self.loss_fn(outputs, mask)
        
        # Get predictions for metrics
        main_pred = outputs['main_pred']
        pred_mask = torch.argmax(main_pred, dim=1)
        
        # Update training metrics
        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(
                pre_image=pred_mask[i].cpu().numpy(),
                gt_image=mask[i].cpu().numpy()
            )
        
        # Log losses
        for key, value in loss_dict.items():
            self.log(f'train_{key}', value, on_step=True, on_epoch=True, prog_bar=False)
        
        return {'loss': loss_dict['total_loss']}
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        img = batch['img']
        mask = batch['gt_semantic_seg']
        
        # Forward pass
        outputs = self.forward(img)
        
        # Compute losses
        loss_dict = self.loss_fn(outputs, mask)
        
        # Get predictions for metrics
        main_pred = outputs['main_pred']
        pred_mask = torch.argmax(main_pred, dim=1)
        
        # Update validation metrics
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(
                pre_image=pred_mask[i].cpu().numpy(),
                gt_image=mask[i].cpu().numpy()
            )
        
        # Log losses
        for key, value in loss_dict.items():
            self.log(f'val_{key}', value, on_step=False, on_epoch=True, prog_bar=False)
        
        return {'val_loss': loss_dict['total_loss']}
    
    def on_train_epoch_end(self):
        """End of training epoch."""
        # Compute training metrics
        train_metrics = self._compute_metrics(self.metrics_train, 'train')
        
        # Log training metrics
        for key, value in train_metrics.items():
            self.log(key, value, prog_bar=True)
        
        # Print training results
        print(f'\nEpoch {self.current_epoch} Training Results:')
        print(f"mIoU: {train_metrics['train_mIoU']:.4f}, F1: {train_metrics['train_F1']:.4f}, OA: {train_metrics['train_OA']:.4f}")
        
        # Reset metrics
        self.metrics_train.reset()
    
    def on_validation_epoch_end(self):
        """End of validation epoch."""
        # Compute validation metrics
        val_metrics = self._compute_metrics(self.metrics_val, 'val')
        
        # Log validation metrics
        for key, value in val_metrics.items():
            self.log(key, value, prog_bar=True)
        
        # Update best validation mIoU
        current_miou = val_metrics['val_mIoU']
        if current_miou > self.best_val_miou:
            self.best_val_miou = current_miou
            self.log('best_val_mIoU', self.best_val_miou)
        
        # Print validation results
        print(f'Epoch {self.current_epoch} Validation Results:')
        print(f"mIoU: {val_metrics['val_mIoU']:.4f}, F1: {val_metrics['val_F1']:.4f}, OA: {val_metrics['val_OA']:.4f}")
        print(f"Best mIoU: {self.best_val_miou:.4f}")
        
        # Print class-wise IoU
        iou_per_class = self.metrics_val.Intersection_over_Union()
        class_names = self.config.get('class_names', [f'Class_{i}' for i in range(len(iou_per_class))])
        print("Class-wise IoU:")
        for name, iou in zip(class_names, iou_per_class):
            print(f"  {name}: {iou:.4f}")
        
        # Reset metrics
        self.metrics_val.reset()
    
    def _compute_metrics(self, evaluator: EnhancedEvaluator, prefix: str) -> Dict[str, float]:
        """Compute metrics from evaluator."""
        dataset_type = self.config.get('dataset_type', 'loveda')
        
        # Get raw metrics
        iou_per_class = evaluator.Intersection_over_Union()
        f1_per_class = evaluator.F1()
        oa = evaluator.OA()
        
        # Compute mean metrics based on dataset type
        if dataset_type in ['vaihingen', 'potsdam']:
            # Exclude background class for aerial datasets
            miou = np.nanmean(iou_per_class[:-1])
            f1 = np.nanmean(f1_per_class[:-1])
        else:
            # Include all classes for other datasets
            miou = np.nanmean(iou_per_class)
            f1 = np.nanmean(f1_per_class)
        
        return {
            f'{prefix}_mIoU': miou,
            f'{prefix}_F1': f1,
            f'{prefix}_OA': oa
        }
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Optimizer
        optimizer_config = self.config.get('optimizer_config', {})
        optimizer_type = optimizer_config.get('type', 'AdamW')
        
        if optimizer_type == 'AdamW':
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=optimizer_config.get('weight_decay', 2.5e-4),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type == 'SGD':
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # Scheduler
        scheduler_config = self.config.get('scheduler_config', {})
        scheduler_type = scheduler_config.get('type', 'CosineAnnealingLR')
        
        if scheduler_type == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config.get('max_epochs', 200),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        elif scheduler_type == 'OneCycleLR':
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=scheduler_config.get('pct_start', 0.3),
                anneal_strategy='cos'
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        else:
            return optimizer
    
    def train_dataloader(self):
        """Get training dataloader."""
        return self.config['train_loader']
    
    def val_dataloader(self):
        """Get validation dataloader."""
        return self.config['val_loader']


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced UNetMamba Training')
    parser.add_argument('-c', '--config_path', type=Path, required=True,
                       help='Path to the configuration file')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode with reduced data')
    return parser.parse_args()


def main():
    """Main training function."""
    args = get_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Update config with command line arguments
    if args.debug:
        config['max_epochs'] = 2
        config['limit_train_batches'] = 10
        config['limit_val_batches'] = 5
    
    # Initialize model
    model = EnhancedSupervisionTrain(config)
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor=config.get('monitor', 'val_mIoU'),
        mode=config.get('monitor_mode', 'max'),
        save_top_k=config.get('save_top_k', 3),
        save_last=config.get('save_last', True),
        dirpath=config.get('weights_path', 'model_weights'),
        filename=config.get('weights_name', 'enhanced_unetmamba') + '-{epoch:02d}-{val_mIoU:.4f}',
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    if config.get('use_early_stopping', True):
        early_stop_callback = EarlyStopping(
            monitor=config.get('monitor', 'val_mIoU'),
            mode=config.get('monitor_mode', 'max'),
            patience=config.get('early_stop_patience', 15),
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Setup loggers
    loggers = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir='lightning_logs',
        name=config.get('log_name', 'enhanced_unetmamba'),
        version=None
    )
    loggers.append(tb_logger)
    
    # CSV logger
    csv_logger = CSVLogger(
        save_dir='lightning_logs',
        name=config.get('log_name', 'enhanced_unetmamba')
    )
    loggers.append(csv_logger)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.get('max_epochs', 200),
        devices=config.get('gpus', 1),
        accelerator='auto',
        strategy='auto',
        callbacks=callbacks,
        logger=loggers,
        check_val_every_n_epoch=config.get('check_val_every_n_epoch', 1),
        gradient_clip_val=config.get('gradient_clip_val', 1.0),
        precision=config.get('precision', '16-mixed'),
        limit_train_batches=config.get('limit_train_batches', 1.0),
        limit_val_batches=config.get('limit_val_batches', 1.0),
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Load from checkpoint if specified
    if args.resume_from and Path(args.resume_from).exists():
        print(f"Resuming training from checkpoint: {args.resume_from}")
        model = EnhancedSupervisionTrain.load_from_checkpoint(
            args.resume_from, config=config
        )
    
    # Start training
    print("Starting Enhanced UNetMamba training...")
    print(f"Model info: {model.net.get_model_info()}")
    
    trainer.fit(
        model=model,
        ckpt_path=args.resume_from if args.resume_from and Path(args.resume_from).exists() else None
    )
    
    # Print best results
    print(f"\nTraining completed!")
    print(f"Best validation mIoU: {model.best_val_miou:.4f}")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
