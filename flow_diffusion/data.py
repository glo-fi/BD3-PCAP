"""
Flow statistics dataset and tokenization utilities for block diffusion.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer
import pickle
import os


class FlowTokenizer:
    """
    Tokenises numerical flow statistics into discrete tokens for block diffusion.
    Uses binning strategy to convert continuous values to discrete tokens.
    """
    
    def __init__(self, n_bins: int = 256, strategy: str = 'quantile', mask_token_id: int = None):
        """
        Args:
            n_bins: Number of bins for each feature
            strategy: Binning strategy ('uniform', 'quantile', 'kmeans')
            mask_token_id: Token ID for mask token (if None, uses n_bins)
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.mask_token_id = mask_token_id if mask_token_id is not None else n_bins
        self.vocab_size = n_bins + 1  # +1 for mask token
        
        self.discretizers = {}
        self.feature_names = []
        self.is_fitted = False
        
    def fit(self, data: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fit the tokeniser on flow statistics data.
        
        Args:
            data: Array of shape (n_samples, n_features)
            feature_names: Names of features
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(data.shape[1])]
        
        self.feature_names = feature_names
        self.actual_n_bins = {}  # Track actual number of bins per feature
        
        # Fit discretizer for each feature
        for i, feature_name in enumerate(feature_names):
            discretizer = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode='ordinal',
                strategy=self.strategy
            )
            discretizer.fit(data[:, i].reshape(-1, 1))
            self.discretizers[feature_name] = discretizer
            
            # Track actual number of bins (may be less than n_bins due to bin removal)
            self.actual_n_bins[feature_name] = discretizer.n_bins_[0]
            
        # Update vocab_size based on maximum actual bins + mask token
        max_actual_bins = max(self.actual_n_bins.values())
        self.vocab_size = max_actual_bins + 1  # +1 for mask token
        self.mask_token_id = max_actual_bins  # Set mask token to max value
            
        self.is_fitted = True
        
    def tokenize(self, data: np.ndarray) -> torch.Tensor:
        """
        Convert flow statistics to tokens.
        
        Args:
            data: Array of shape (n_samples, n_features)
            
        Returns:
            Tensor of shape (n_samples, n_features) with token IDs
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted first")
            
        tokens = []
        for i, feature_name in enumerate(self.feature_names):
            discretizer = self.discretizers[feature_name]
            feature_tokens = discretizer.transform(data[:, i].reshape(-1, 1)).flatten()
            tokens.append(feature_tokens)
            
        return torch.tensor(np.column_stack(tokens), dtype=torch.long)
    
    def detokenize(self, tokens: torch.Tensor) -> np.ndarray:
        """
        Convert tokens back to flow statistics (approximate).
        
        Args:
            tokens: Tensor of shape (n_samples, n_features) with token IDs
            
        Returns:
            Array of shape (n_samples, n_features) with reconstructed values
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted first")
            
        tokens_np = tokens.cpu().numpy()
        reconstructed = []
        
        for i, feature_name in enumerate(self.feature_names):
            discretizer = self.discretizers[feature_name]
            feature_tokens = tokens_np[:, i].reshape(-1, 1)
            
            # Get actual number of bins for this feature
            actual_bins = self.actual_n_bins[feature_name]
            
            # Handle mask tokens and out-of-range tokens by replacing with random valid tokens
            mask_indices = feature_tokens.flatten() == self.mask_token_id
            out_of_range_indices = (feature_tokens.flatten() >= actual_bins) | (feature_tokens.flatten() < 0)
            invalid_indices = mask_indices | out_of_range_indices
            
            if invalid_indices.any():
                # Replace invalid tokens with random valid tokens for this specific feature
                feature_tokens[invalid_indices] = np.random.randint(0, actual_bins, size=invalid_indices.sum()).reshape(-1, 1)
            
            # Ensure all tokens are in valid range [0, actual_bins-1] for this feature
            feature_tokens = np.clip(feature_tokens, 0, actual_bins - 1)
            
            # Double-check before inverse transform
            assert feature_tokens.min() >= 0, f"Negative token found: {feature_tokens.min()}"
            assert feature_tokens.max() < actual_bins, f"Token >= actual_bins found: {feature_tokens.max()} >= {actual_bins}"
            
            # Inverse transform (returns bin centers)
            reconstructed_feature = discretizer.inverse_transform(feature_tokens).flatten()
            reconstructed.append(reconstructed_feature)
            
        return np.column_stack(reconstructed)
    
    def save(self, path: str):
        """Save tokeniser to file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, path: str):
        """Load tokeniser from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class FlowDataset(Dataset):
    """
    Dataset class for network flow statistics.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        tokenizer: FlowTokenizer,
        sequence_length: int = 128,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Args:
            data: Flow statistics array of shape (n_flows, n_features)
            tokenizer: Fitted FlowTokenizer
            sequence_length: Maximum sequence length for padding
            feature_names: Names of features
        """
        self.data = data
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.feature_names = feature_names or [f"feature_{i}" for i in range(data.shape[1])]
        
        # Tokenize the data
        self.tokens = self.tokenizer.tokenize(data)
        
        # Pad sequences to fixed length (each flow is one sequence)
        if self.tokens.shape[1] < sequence_length:
            padding = torch.full(
                (self.tokens.shape[0], sequence_length - self.tokens.shape[1]),
                fill_value=self.tokenizer.mask_token_id,
                dtype=torch.long
            )
            self.tokens = torch.cat([self.tokens, padding], dim=1)
        elif self.tokens.shape[1] > sequence_length:
            self.tokens = self.tokens[:, :sequence_length]
            
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        return self.tokens[idx]
    
    def get_vocab_size(self):
        return self.tokenizer.vocab_size
    
    def get_mask_token_id(self):
        return self.tokenizer.mask_token_id


def create_sample_flow_data(n_flows: int = 10000, n_features: int = 8) -> Tuple[np.ndarray, List[str]]:
    """
    Create sample network flow statistics data for testing.
    
    Args:
        n_flows: Number of flow samples
        n_features: Number of features per flow
        
    Returns:
        Tuple of (data array, feature names)
    """
    np.random.seed(42)
    
    # Define realistic flow features
    feature_names = [
        'duration',           # Flow duration in seconds
        'total_packets',      # Total packets in flow
        'total_bytes',        # Total bytes in flow
        'packets_per_sec',    # Packets per second
        'bytes_per_sec',      # Bytes per second
        'avg_packet_size',    # Average packet size
        'protocol',           # Protocol (6=TCP, 17=UDP, 1=ICMP)
        'port_ratio'          # Ratio of source to dest port
    ][:n_features]
    
    data = []
    
    for _ in range(n_flows):
        # Generate correlated flow statistics
        duration = np.random.exponential(10.0)  # Duration in seconds
        total_packets = np.random.poisson(max(1, duration * 5))  # Packets related to duration
        avg_packet_size = np.random.normal(800, 300)  # Average packet size
        avg_packet_size = max(64, min(1500, avg_packet_size))  # Clamp to realistic range
        
        total_bytes = total_packets * avg_packet_size + np.random.normal(0, total_packets * 50)
        total_bytes = max(total_packets * 64, total_bytes)  # Minimum realistic bytes
        
        packets_per_sec = total_packets / max(duration, 0.001)
        bytes_per_sec = total_bytes / max(duration, 0.001)
        
        # Protocol distribution: 70% TCP, 25% UDP, 5% ICMP
        protocol_prob = np.random.random()
        if protocol_prob < 0.7:
            protocol = 6  # TCP
        elif protocol_prob < 0.95:
            protocol = 17  # UDP
        else:
            protocol = 1  # ICMP
            
        port_ratio = np.random.exponential(1.0)  # Port ratio
        
        flow_features = [
            duration,
            total_packets,
            total_bytes,
            packets_per_sec,
            bytes_per_sec,
            avg_packet_size,
            protocol,
            port_ratio
        ][:n_features]
        
        data.append(flow_features)
    
    return np.array(data), feature_names


def load_flow_data(file_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load flow statistics from CSV file.
    
    Args:
        file_path: Path to CSV file with flow statistics
        
    Returns:
        Tuple of (data array, feature names)
    """
    df = pd.read_csv(file_path)
    feature_names = df.columns.tolist()
    data = df.values
    return data, feature_names


def create_flow_dataloader(
    data: np.ndarray,
    feature_names: List[str],
    batch_size: int = 32,
    sequence_length: int = 8,
    train_split: float = 0.8,
    n_bins: int = 256,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, FlowTokenizer]:
    """
    Create train/val dataloaders for flow statistics.
    
    Args:
        data: Flow statistics array
        feature_names: Feature names
        batch_size: Batch size
        sequence_length: Sequence length (number of features per flow)
        train_split: Training split ratio
        n_bins: Number of bins for tokenization
        
    Returns:
        Tuple of (train_loader, val_loader, tokenizer)
    """
    # Split data
    n_train = int(len(data) * train_split)
    train_data = data[:n_train]
    val_data = data[n_train:]
    
    # Fit tokenizer on training data
    tokenizer = FlowTokenizer(n_bins=n_bins)
    tokenizer.fit(train_data, feature_names)
    
    # Create datasets
    train_dataset = FlowDataset(train_data, tokenizer, sequence_length, feature_names)
    val_dataset = FlowDataset(val_data, tokenizer, sequence_length, feature_names)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, tokenizer
