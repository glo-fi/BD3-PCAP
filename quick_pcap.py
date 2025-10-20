#!/usr/bin/env python3
"""
Quickrun script for Flow Block Diffusion Model trained on real PCAP data.
Processes PCAP file, extracts packet features, trains a model, and generates new packets.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import Dict, Any, List
import argparse

# Add the project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flow_diffusion.data import create_flow_dataloader, FlowTokenizer
from flow_diffusion.model import FlowBD3LM, compute_diffusion_loss
from flow_diffusion.train import FlowTrainer
from flow_diffusion.evaluation import FlowEvaluator, generate_flows
from flow_diffusion.pcap_reader import process_pcap


def create_quick_pcap_config(n_packets: int) -> Dict[str, Any]:
    """Create configuration based on PCAP data size."""
    return {
        # Data parameters
        "n_packets": n_packets,
        "n_bins": 128,
        "batch_size": 32,
        "train_split": 0.8,
        # Model parameters
        "d_model": 512,
        "n_heads": 16,
        "n_layers": 16,
        "d_ff": 1024,
        "dropout": 0.1,
        "block_size": 1,
        # Training parameters
        "epochs": 3,
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "mask_prob": 0.15,
        # Output
        "output_dir": "./quick_pcap_outputs",
        "save_every": 10,
    }


def select_numeric_features(
    df: pd.DataFrame, variance_threshold: float = 1e-4
) -> tuple[pd.DataFrame, List[str]]:
    """
    Select only numeric features suitable for training.
    Filters out non-numeric columns like IP addresses and protocol names.
    Also removes constant or near-constant features.

    Args:
        df: DataFrame with all packet features
        variance_threshold: Minimum variance required to keep a feature

    Returns:
        Tuple of (numeric DataFrame, list of feature names)
    """
    # Features to explicitly exclude (non-numeric or identifiers)
    exclude_features = [
        "src_ip",
        "dst_ip",  # IP addresses
        "protocol",  # String protocol name
        "timestamp",  # Will be handled separately if needed
    ]

    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove excluded features
    selected_features = [col for col in numeric_cols if col not in exclude_features]

    # Create filtered DataFrame
    numeric_df = df[selected_features].copy()

    # Handle NaN values - replace with 0
    numeric_df = numeric_df.fillna(0)

    # Handle inf values
    numeric_df = numeric_df.replace([np.inf, -np.inf], 0)

    # Remove constant features (variance == 0)
    variances = numeric_df.var()
    non_constant_features = variances[variances > 0].index.tolist()

    if len(non_constant_features) < len(selected_features):
        constant_features = set(selected_features) - set(non_constant_features)
        print(f"   Removed {len(constant_features)} constant features: {constant_features}")

    numeric_df = numeric_df[non_constant_features]

    # Also check for features with very limited unique values
    # If a feature has fewer unique values than we want bins, it will cause issues
    min_unique_values = 10  # Minimum unique values for meaningful binning

    valid_features = []
    low_variance_features = []

    for feature in non_constant_features:
        n_unique = numeric_df[feature].nunique()
        variance = numeric_df[feature].var()

        # Keep feature if it has enough unique values and sufficient variance
        if n_unique >= min_unique_values and variance > variance_threshold:
            valid_features.append(feature)
        else:
            low_variance_features.append(feature)

    if low_variance_features:
        print(f"   Removed {len(low_variance_features)} low-variance/low-cardinality features: {low_variance_features}")

    numeric_df = numeric_df[valid_features]
    selected_features = valid_features

    return numeric_df, selected_features


def main():
    parser = argparse.ArgumentParser(
        description="Train Flow Block Diffusion Model on PCAP data"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input PCAP file"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help='BPF filter for packet filtering (e.g., "tcp port 443")',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./quick_pcap_outputs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max_packets",
        type=int,
        default=None,
        help="Maximum number of packets to process (default: all)",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Flow Block Diffusion Model - PCAP Training")
    print("=" * 60)

    # Step 1: Process PCAP file
    print("\n1. Processing PCAP file...")
    print(f"   Input file: {args.input}")
    if args.filter:
        print(f"   BPF filter: {args.filter}")

    # Process PCAP and get feature vectors
    feature_vectors = process_pcap(
        input_file=args.input,
        bpf_filter=args.filter,
        output_file=None,  # Don't write to CSV during processing
        flush_interval=None,
    )

    if not feature_vectors:
        print("Error: No packets were processed. Check your PCAP file and filter.")
        return 1

    print(f"   Processed {len(feature_vectors)} packets")

    # Limit packets if requested
    if args.max_packets and len(feature_vectors) > args.max_packets:
        print(f"   Limiting to {args.max_packets} packets")
        feature_vectors = feature_vectors[: args.max_packets]

    # Step 2: Convert to DataFrame and prepare features
    print("\n2. Preparing features for training...")
    df = pd.DataFrame(feature_vectors)

    print(f"   Raw features: {len(df.columns)}")
    print(f"   Feature names: {list(df.columns)}")

    # Select only numeric features
    numeric_df, feature_names = select_numeric_features(df)

    print(f"   Selected {len(feature_names)} numeric features for training")
    print(f"   Training features: {feature_names}")

    # Convert to numpy array
    data = numeric_df.values

    if data.shape[0] < 100:
        print(
            f"\nWarning: Only {data.shape[0]} packets available. Consider using a larger PCAP file."
        )

    # Display statistics
    print(f"\n   Dataset shape: {data.shape}")
    print(f"   Feature statistics:")
    print(numeric_df.describe())

    # Step 3: Create configuration
    config = create_quick_pcap_config(len(data))
    config["output_dir"] = args.output_dir
    config["epochs"] = args.epochs
    config["n_features"] = len(feature_names)
    config["sequence_length"] = len(feature_names)

    print(f"\n3. Configuration:")
    print(
        f"   Model: {config['n_layers']} layers, {config['n_heads']} heads, {config['d_model']} dim"
    )
    print(f"   Training: {config['epochs']} epochs, batch size {config['batch_size']}")
    print(f"   Output: {config['output_dir']}")

    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)

    # Save raw packet data for reference
    df.to_csv(os.path.join(config["output_dir"], "original_packets.csv"), index=False)
    numeric_df.to_csv(
        os.path.join(config["output_dir"], "training_features.csv"), index=False
    )

    # Save configuration
    with open(os.path.join(config["output_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Step 4: Create data loaders and tokenizer
    print("\n4. Creating data loaders and tokenizer...")
    train_loader, val_loader, tokenizer = create_flow_dataloader(
        data=data,
        feature_names=feature_names,
        batch_size=config["batch_size"],
        sequence_length=config["sequence_length"],
        train_split=config["train_split"],
        n_bins=config["n_bins"],
    )

    print(f"   Tokenizer vocabulary size: {tokenizer.vocab_size}")
    print(f"   Mask token ID: {tokenizer.mask_token_id}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")

    # Save tokenizer
    tokenizer.save(os.path.join(config["output_dir"], "tokenizer.pkl"))

    # Step 5: Create model
    print("\n5. Creating model...")
    model = FlowBD3LM(
        vocab_size=tokenizer.vocab_size,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["sequence_length"],
        dropout=config["dropout"],
        mask_token_id=tokenizer.mask_token_id,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model created with {total_params:,} parameters")

    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"   Using device: {device}")

    # Get a sample batch
    sample_batch = next(iter(train_loader)).to(device)
    print(f"   Sample batch shape: {sample_batch.shape}")

    # Test loss computation
    with torch.no_grad():
        loss = compute_diffusion_loss(
            model,
            sample_batch,
            block_size=config["block_size"],
            mask_prob=config["mask_prob"],
        )
        print(f"   Initial loss: {loss.item():.4f}")

    # Step 6: Train the model
    print("\n6. Training model...")
    trainer = FlowTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        config=config,
    )

    trainer.train()

    print("\nTraining completed!")

    # Step 7: Generate new packet features
    print("\n7. Generating new packet features...")
    n_generated = min(1000, len(data))

    # Load best model
    best_checkpoint = torch.load(os.path.join(config["output_dir"], "best.pt"))
    model.load_state_dict(best_checkpoint["model_state_dict"])

    # Generate samples
    generated_data = generate_flows(
        model=model,
        tokenizer=tokenizer,
        n_samples=n_generated,
        block_size=config["block_size"],
        device=device,
    )

    print(f"   Generated {len(generated_data)} new packet samples")
    print(f"   Generated data shape: {generated_data.shape}")

    # Save generated data
    generated_df = pd.DataFrame(generated_data, columns=feature_names)
    generated_df.to_csv(
        os.path.join(config["output_dir"], "generated_packets.csv"), index=False
    )

    # Step 8: Evaluate the results
    print("\n8. Evaluating generated packets...")
    evaluator = FlowEvaluator(feature_names)

    # Use test split for evaluation
    n_test = int(len(data) * (1 - config["train_split"]))
    test_data = data[-n_test:]

    results = evaluator.evaluate(
        real_data=test_data,
        generated_data=generated_data,
        output_dir=config["output_dir"],
    )

    # Print key metrics
    print("\n   Evaluation Results:")
    print("   " + "-" * 30)
    print(f"   Mean KS statistic: {results['mean_ks_stat']:.4f}")
    print(f"   Mean Earth Mover Distance: {results['mean_emd']:.4f}")
    print(f"   Correlation Frobenius norm: {results['corr_frobenius_norm']:.4f}")
    print(f"   Mean distance to real data: {results['mean_distance_to_real']:.4f}")

    # Save detailed results
    with open(os.path.join(config["output_dir"], "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Step 9: Create comparison visualizations
    print("\n9. Creating comparison visualizations...")

    # Feature distributions
    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i, feature_name in enumerate(feature_names):
        if i < len(axes):
            ax = axes[i]
            ax.hist(
                test_data[:, i],
                bins=30,
                alpha=0.7,
                label="Real",
                density=True,
                color="blue",
            )
            ax.hist(
                generated_data[:, i],
                bins=30,
                alpha=0.7,
                label="Generated",
                density=True,
                color="orange",
            )
            ax.set_title(f"{feature_name}", fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)

    # Remove extra subplots
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(os.path.join(config["output_dir"], "comparison.png"), dpi=150)
    plt.close()

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Processed {len(feature_vectors)} packets from PCAP file")
    print(f"Extracted {len(feature_names)} numeric features")
    print(
        f"Trained model with {total_params:,} parameters for {config['epochs']} epochs"
    )
    print(f"Generated {n_generated} new packet samples")
    print(f"Evaluated generation quality with multiple metrics")
    print(f"\nResults saved to: {config['output_dir']}")
    print("\nKey files created:")
    print(f"  - {config['output_dir']}/best.pt (trained model)")
    print(f"  - {config['output_dir']}/tokenizer.pkl (tokenizer)")
    print(f"  - {config['output_dir']}/original_packets.csv (original PCAP features)")
    print(
        f"  - {config['output_dir']}/training_features.csv (numeric features used for training)"
    )
    print(
        f"  - {config['output_dir']}/generated_packets.csv (generated packet features)"
    )
    print(f"  - {config['output_dir']}/evaluation_results.json (evaluation metrics)")
    print(f"  - {config['output_dir']}/comparison.png (comparison plots)")
    print(
        f"  - {config['output_dir']}/feature_distributions.png (detailed distributions)"
    )
    print(f"  - {config['output_dir']}/correlation_matrices.png (correlation analysis)")
    print(
        "\nThe model can now generate realistic packet features from real network traffic!"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
