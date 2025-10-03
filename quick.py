#!/usr/bin/env python3
"""
Quickrun script for Flow Block Diffusion Model.
Creates sample data, trains a small model, and generates new flow statistics.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, Any

# Add the project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flow_diffusion.data import create_sample_flow_data, create_flow_dataloader, FlowTokenizer
from flow_diffusion.model import FlowBD3LM, compute_diffusion_loss
from flow_diffusion.train import FlowTrainer
from flow_diffusion.evaluation import FlowEvaluator, generate_flows


def create_quick_config() -> Dict[str, Any]:
    """Create configuration."""
    return {
        # Data parameters
        'n_flows': 5000,
        'n_features': 6, 
        'sequence_length': 6,
        'n_bins': 128, 
        'batch_size': 32,
        'train_split': 0.8,
        
        # Model parameters 
        'd_model': 512,
        'n_heads': 16,
        'n_layers': 16,
        'd_ff': 1024,
        'dropout': 0.1,
        'block_size': 1,
        
        # Training parameters 
        'epochs': 10,  
        'learning_rate': 3e-5,
        'weight_decay': 0.01,
        'gradient_clip': 1.0,
        'mask_prob': 0.15,
        
        # Output
        'output_dir': './quick_outputs',
        'save_every': 10,
    }


def main():
    print("=" * 60)
    print("Flow Block Diffusion Model")
    print("=" * 60)
    
    # Create configuration
    config = create_quick_config()
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Create sample flow data
    print("\n1. Creating sample flow statistics data...")
    data, feature_names = create_sample_flow_data(
        config['n_flows'],
        config['n_features']
    )
    
    print(f"Dataset: {len(data)} flows, {len(feature_names)} features")
    print(f"Features: {feature_names}")
    print(f"Data shape: {data.shape}")
    print(f"Sample statistics:\n{np.array(data)}")
    
    # Create data loaders and tokenizer
    print("\n2. Creating data loaders and tokenizer...")
    train_loader, val_loader, tokenizer = create_flow_dataloader(
        data=data,
        feature_names=feature_names,
        batch_size=config['batch_size'],
        sequence_length=config['sequence_length'],
        train_split=config['train_split'],
        n_bins=config['n_bins'],
    )
    
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    print(f"Mask token ID: {tokenizer.mask_token_id}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Save tokenizer
    tokenizer.save(os.path.join(config['output_dir'], 'tokenizer.pkl'))
    
    # Create and inspect model
    print("\n3. Creating model...")
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
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Get a sample batch
    sample_batch = next(iter(train_loader)).to(device)
    print(f"Sample batch shape: {sample_batch.shape}")
    
    # Test loss computation
    with torch.no_grad():
        loss = compute_diffusion_loss(
            model,
            sample_batch,
            block_size=config['block_size'],
            mask_prob=config['mask_prob']
        )
        print(f"Sample loss: {loss.item():.4f}")
    
    # Train the model
    print("\n4. Training model...")
    trainer = FlowTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        config=config,
    )
    
    trainer.train()
    
    print("Training completed!")
    
    # Step 5: Generate new flow statistics
    print("\n5. Generating new flow statistics...")
    n_generated = 1000
    
    # Load best model
    best_checkpoint = torch.load(os.path.join(config['output_dir'], 'best.pt'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Generate samples
    generated_data = generate_flows(
        model=model,
        tokenizer=tokenizer,
        n_samples=n_generated,
        block_size=config['block_size'],
        device=device
    )
    
    print(f"Generated {len(generated_data)} new flow samples")
    print(f"Generated data shape: {generated_data.shape}")
    
    # Step 6: Evaluate the results
    print("\n6. Evaluating generated flows...")
    evaluator = FlowEvaluator(feature_names)
    
    # Use test split for evaluation
    n_test = int(len(data) * (1 - config['train_split']))
    test_data = data[-n_test:]
    
    results = evaluator.evaluate(
        real_data=test_data,
        generated_data=generated_data,
        output_dir=config['output_dir']
    )
    
    # Print key metrics
    print("\nEvaluation Results:")
    print("-" * 30)
    print(f"Mean KS statistic: {results['mean_ks_stat']:.4f}")
    print(f"Mean Earth Mover Distance: {results['mean_emd']:.4f}")
    print(f"Correlation Frobenius norm: {results['corr_frobenius_norm']:.4f}")
    print(f"Mean distance to real data: {results['mean_distance_to_real']:.4f}")
    
    # Save detailed results
    with open(os.path.join(config['output_dir'], 'quick_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Step 7: Create simple comparison plot
    print("\n7. Creating comparison visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature_name in enumerate(feature_names):
        if i < len(axes):
            ax = axes[i]
            ax.hist(test_data[:, i], bins=30, alpha=0.7, label='Real', density=True)
            ax.hist(generated_data[:, i], bins=30, alpha=0.7, label='Generated', density=True)
            ax.set_title(f'{feature_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'comparison.png'), dpi=150)
    plt.close()
    
    print(f"Run completed! Results saved to: {config['output_dir']}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Created {config['n_flows']} sample flows with {config['n_features']} features")
    print(f"Trained model with {total_params:,} parameters for {config['epochs']} epochs")
    print(f"Generated {n_generated} new flow samples")
    print(f"Evaluated generation quality with multiple metrics")
    print(f"Created visualizations and saved results to {config['output_dir']}")
    print("\nKey files created:")
    print(f"  - {config['output_dir']}/best.pt (trained model)")
    print(f"  - {config['output_dir']}/tokenizer.pkl (tokenizer)")
    print(f"  - {config['output_dir']}/results.json (evaluation metrics)")
    print(f"  - {config['output_dir']}/comparison.png (comparison plots)")
    print(f"  - {config['output_dir']}/feature_distributions.png (detailed distributions)")
    print("\nThe model can now generate realistic network flow statistics!")


if __name__ == '__main__':
    main()
