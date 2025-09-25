#!/usr/bin/env python3
"""
Simple CLI tool for generating flow statistics using trained BD3-LM model.
"""

import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
import json

# Add the project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flow_diffusion.data import FlowTokenizer
from flow_diffusion.model import FlowBD3LM
from flow_diffusion.evaluation import generate_flows


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Generate network flow statistics using trained BD3-LM')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to tokenizer file')
    parser.add_argument('--output', type=str, default='generated_flows.csv', help='Output CSV file')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling parameter')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = FlowTokenizer.load(args.tokenizer)
    print(f"Tokenizer loaded. Vocabulary size: {tokenizer.vocab_size}")
    
    # Load model
    print(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=device)
    config = checkpoint['config']
    
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Parameters: {total_params:,}")
    
    # Generate samples
    print(f"Generating {args.n_samples} flow samples...")
    generated_data = generate_flows(
        model=model,
        tokenizer=tokenizer,
        n_samples=args.n_samples,
        block_size=config.get('block_size', 1),
        temperature=args.temperature,
        top_p=args.top_p,
        device=device
    )
    
    # Save to CSV
    feature_names = tokenizer.feature_names
    df = pd.DataFrame(generated_data, columns=feature_names)
    df.to_csv(args.output, index=False)
    
    print(f"Generated flows saved to {args.output}")
    print(f"Generated {len(generated_data)} samples with {len(feature_names)} features")
    print(f"Features: {feature_names}")
    
    # Print some statistics
    print("\nGenerated data statistics:")
    print(df.describe())


if __name__ == '__main__':
    main()