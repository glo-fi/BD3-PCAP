"""
Training script for Flow Block Diffusion Model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import os
import json
from typing import Dict, Any, Tuple

from .model import FlowBD3LM, compute_diffusion_loss
from .data import create_sample_flow_data, create_flow_dataloader, FlowTokenizer
from .evaluation import FlowEvaluator


class FlowTrainer:
    """Trainer class for Flow BD3-LM."""
    
    def __init__(
        self,
        model: FlowBD3LM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: FlowTokenizer,
        config: Dict[str, Any],
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs']
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Create output directory
        os.makedirs(config['output_dir'], exist_ok=True)
        
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch in progress_bar:
            input_ids = batch.to(self.device)
            
            # Compute loss
            loss = compute_diffusion_loss(
                self.model,
                input_ids,
                block_size=self.config['block_size'],
                mask_prob=self.config['mask_prob']
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch.to(self.device)
                
                loss = compute_diffusion_loss(
                    self.model,
                    input_ids,
                    block_size=self.config['block_size'],
                    mask_prob=self.config['mask_prob']
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.config['output_dir'], 'latest.pt'))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.config['output_dir'], 'best.pt'))
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
    
    def plot_losses(self):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.config['output_dir'], 'losses.png'))
        plt.close()
    
    def train(self):
        """Main training loop."""
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(is_best)
            
            # Print progress
            print(f"Epoch {epoch + 1}/{self.config['epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Best Val Loss: {self.best_val_loss:.4f}")
            print("-" * 50)
            
            # Plot losses
            if (epoch + 1) % 10 == 0:
                self.plot_losses()
        
        print("Training completed!")
        self.plot_losses()


def create_default_config() -> Dict[str, Any]:
    """Create default training configuration."""
    return {
        # Data parameters
        'n_flows': 10000,
        'n_features': 8,
        'sequence_length': 8,
        'n_bins': 256,
        'batch_size': 64,
        'train_split': 0.8,
        
        # Model parameters
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 1024,
        'dropout': 0.1,
        'block_size': 1,
        
        # Training parameters
        'epochs': 100,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'gradient_clip': 1.0,
        'mask_prob': 0.15,
        
        # Output
        'output_dir': './outputs',
        'save_every': 10,
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Flow Block Diffusion Model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data', type=str, help='Path to flow data CSV')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = create_default_config()
    if args.config:
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Create or load data
    if args.data:
        print(f"Loading data from {args.data}")
        from .data import load_flow_data
        data, feature_names = load_flow_data(args.data)
    else:
        print("Creating sample flow data")
        data, feature_names = create_sample_flow_data(
            config['n_flows'],
            config['n_features']
        )
    
    print(f"Dataset: {len(data)} flows, {len(feature_names)} features")
    print(f"Features: {feature_names}")
    
    # Create data loaders
    train_loader, val_loader, tokenizer = create_flow_dataloader(
        data=data,
        feature_names=feature_names,
        batch_size=config['batch_size'],
        sequence_length=config['sequence_length'],
        train_split=config['train_split'],
        n_bins=config['n_bins'],
    )
    
    # Save tokenizer
    tokenizer.save(os.path.join(config['output_dir'], 'tokenizer.pkl'))
    
    # Create model
    model = FlowBD3LM(
        vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['sequence_length'],
        dropout=config['dropout'],
        mask_token_id=tokenizer.mask_token_id,
    )
    
    # Create trainer
    trainer = FlowTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        config=config,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Save configuration
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train the model
    trainer.train()


if __name__ == '__main__':
    main()