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
from flow_diffusion.pcap_reader import process_pcap


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Generate network flow statistics using trained BD3-LM or process PCAP files"
    )

    # Add subcommands for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")

    # Generate mode (original functionality)
    generate_parser = subparsers.add_parser(
        "generate", help="Generate synthetic flows using trained model"
    )
    generate_parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model checkpoint"
    )
    generate_parser.add_argument(
        "--tokenizer", type=str, required=True, help="Path to tokenizer file"
    )
    generate_parser.add_argument(
        "--output", type=str, default="generated_flows.csv", help="Output CSV file"
    )
    generate_parser.add_argument(
        "--n_samples", type=int, default=1000, help="Number of samples to generate"
    )
    generate_parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    generate_parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p sampling parameter"
    )
    generate_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # PCAP mode (new functionality)
    pcap_parser = subparsers.add_parser(
        "pcap", help="Process PCAP file and extract packet features"
    )
    pcap_parser.add_argument(
        "--input", type=str, required=True, help="Path to input PCAP file"
    )
    pcap_parser.add_argument(
        "--output", type=str, default="output", help="Output directory for CSV files"
    )
    pcap_parser.add_argument(
        "--filter", type=str, default="", help="BPF filter for packet filtering"
    )
    pcap_parser.add_argument(
        "--flush_interval",
        type=int,
        default=1000,
        help="Number of packets after which to flush to CSV",
    )
    # Optional: Generate synthetic flows using trained model
    pcap_parser.add_argument(
        "--model", type=str, default=None, help="Path to trained model checkpoint (optional)"
    )
    pcap_parser.add_argument(
        "--tokenizer", type=str, default=None, help="Path to tokenizer file (optional)"
    )
    pcap_parser.add_argument(
        "--n_samples", type=int, default=1000, help="Number of synthetic samples to generate (if model provided)"
    )
    pcap_parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature (if model provided)"
    )
    pcap_parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p sampling parameter (if model provided)"
    )

    args = parser.parse_args()

    if args.mode == "generate":
        generate_synthetic_flows(args)
    elif args.mode == "pcap":
        process_pcap_file(args)
    else:
        parser.print_help()


def generate_synthetic_flows(args):
    """Generate synthetic flows using trained model."""
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = FlowTokenizer.load(args.tokenizer)
    print(f"Tokenizer loaded. Vocabulary size: {tokenizer.vocab_size}")

    # Load model
    print(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=device)
    config = checkpoint["config"]

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

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Parameters: {total_params:,}")

    # Generate samples
    print(f"Generating {args.n_samples} flow samples...")
    generated_data = generate_flows(
        model=model,
        tokenizer=tokenizer,
        n_samples=args.n_samples,
        block_size=config.get("block_size", 1),
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
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


def process_pcap_file(args):
    """Process PCAP file, extract packet features, and optionally generate synthetic flows."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Set up output file paths
    packet_features_file = os.path.join(args.output, "packet_features.csv")

    print(f"Processing PCAP file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"BPF filter: '{args.filter}'" if args.filter else "No BPF filter applied")

    # Process the PCAP file
    feature_vectors = process_pcap(
        input_file=args.input,
        bpf_filter=args.filter,
        output_file=packet_features_file,
        flush_interval=args.flush_interval,
    )

    print(f"Processed {len(feature_vectors)} packets")
    print(f"Packet features saved to: {packet_features_file}")

    if not feature_vectors:
        print("No packets were processed. Check your PCAP file and filter settings.")
        return

    # Create DataFrame and display statistics
    df = pd.DataFrame(feature_vectors)

    print(f"\nPacket statistics:")
    print(f"Total packets: {len(df)}")
    print(f"Features per packet: {len(df.columns)}")
    print(f"Feature names: {list(df.columns)}")

    # Display basic statistics for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        print(f"\nNumeric feature statistics:")
        print(df[numeric_columns].describe())

    # Display protocol distribution if available
    if "protocol" in df.columns:
        print(f"\nProtocol distribution:")
        print(df["protocol"].value_counts(dropna=False))

    # Display IP version distribution if available
    if "ip_version" in df.columns:
        print(f"\nIP version distribution:")
        version_counts = df["ip_version"].value_counts(dropna=False)
        for version, count in version_counts.items():
            if pd.isna(version):
                print(f"  Non-IP: {count}")
            elif version == 4:
                print(f"  IPv4: {count}")
            elif version == 6:
                print(f"  IPv6: {count}")
            else:
                print(f"  Version {version}: {count}")

    # If model and tokenizer are provided, generate synthetic flows
    if hasattr(args, "model") and args.model and hasattr(args, "tokenizer") and args.tokenizer:
        print(f"\n{'='*60}")
        print("Generating synthetic flows using trained model...")
        print(f"{'='*60}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load tokenizer
        print(f"Loading tokenizer from {args.tokenizer}")
        tokenizer = FlowTokenizer.load(args.tokenizer)
        print(f"Tokenizer loaded. Vocabulary size: {tokenizer.vocab_size}")

        # Load model
        print(f"Loading model from {args.model}")
        checkpoint = torch.load(args.model, map_location=device)
        config = checkpoint["config"]

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

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded. Parameters: {total_params:,}")

        # Convert packet features to the format expected by the model
        # Select only numeric columns that match the tokenizer's feature names
        available_features = [f for f in tokenizer.feature_names if f in df.columns]

        if len(available_features) < len(tokenizer.feature_names):
            print(f"\nWarning: Only {len(available_features)}/{len(tokenizer.feature_names)} features available")
            print(f"Available: {available_features}")
            print(f"Expected: {tokenizer.feature_names}")

            # Fill missing features with zeros or appropriate defaults
            for feature in tokenizer.feature_names:
                if feature not in df.columns:
                    df[feature] = 0.0

        # Extract features in the correct order
        packet_data = df[tokenizer.feature_names].values

        # Handle NaN values
        packet_data = np.nan_to_num(packet_data, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"\nInput data shape: {packet_data.shape}")
        print(f"Features: {tokenizer.feature_names}")

        # Determine number of samples to generate
        n_samples = getattr(args, "n_samples", min(1000, len(packet_data)))
        temperature = getattr(args, "temperature", 1.0)
        top_p = getattr(args, "top_p", 0.9)

        # Generate synthetic flows
        print(f"\nGenerating {n_samples} synthetic flow samples...")
        generated_data = generate_flows(
            model=model,
            tokenizer=tokenizer,
            n_samples=n_samples,
            block_size=config.get("block_size", 1),
            temperature=temperature,
            top_p=top_p,
            device=device,
        )

        # Save generated flows
        generated_flows_file = os.path.join(args.output, "generated_flows.csv")
        feature_names = tokenizer.feature_names
        generated_df = pd.DataFrame(generated_data, columns=feature_names)
        generated_df.to_csv(generated_flows_file, index=False)

        print(f"\nGenerated flows saved to: {generated_flows_file}")
        print(f"Generated {len(generated_data)} samples with {len(feature_names)} features")

        # Print some statistics
        print("\nGenerated flow statistics:")
        print(generated_df.describe())
    else:
        print("\nNote: To generate synthetic flows, provide --model and --tokenizer arguments")


if __name__ == "__main__":
    main()

