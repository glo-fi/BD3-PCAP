"""
Evaluation metrics and utilities for flow generation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Any
import os

from .data import FlowTokenizer
from .model import FlowBD3LM


class FlowEvaluator:
    """Evaluator for generated flow statistics."""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        
    def compute_distribution_metrics(
        self,
        real_data: np.ndarray,
        generated_data: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute distribution comparison metrics.
        
        Args:
            real_data: Real flow data [n_samples, n_features]
            generated_data: Generated flow data [n_samples, n_features]
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        n_features = real_data.shape[1]
        
        # Per-feature metrics
        for i, feature_name in enumerate(self.feature_names):
            real_feat = real_data[:, i]
            gen_feat = generated_data[:, i]
            
            # Statistical tests
            ks_stat, ks_pvalue = stats.ks_2samp(real_feat, gen_feat)
            metrics[f'{feature_name}_ks_stat'] = ks_stat
            metrics[f'{feature_name}_ks_pvalue'] = ks_pvalue
            
            # Moment matching
            metrics[f'{feature_name}_mean_diff'] = abs(np.mean(real_feat) - np.mean(gen_feat))
            metrics[f'{feature_name}_std_diff'] = abs(np.std(real_feat) - np.std(gen_feat))
            
            # Earth mover's distance (Wasserstein)
            emd = stats.wasserstein_distance(real_feat, gen_feat)
            metrics[f'{feature_name}_emd'] = emd
        
        # Overall metrics
        metrics['mean_ks_stat'] = np.mean([metrics[f'{f}_ks_stat'] for f in self.feature_names])
        metrics['mean_emd'] = np.mean([metrics[f'{f}_emd'] for f in self.feature_names])
        
        return metrics
    
    def compute_correlation_metrics(
        self,
        real_data: np.ndarray,
        generated_data: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute correlation structure metrics.
        
        Args:
            real_data: Real flow data
            generated_data: Generated flow data
            
        Returns:
            Dictionary of correlation metrics
        """
        real_corr = np.corrcoef(real_data.T)
        gen_corr = np.corrcoef(generated_data.T)
        
        # Correlation matrix differences
        corr_diff = np.abs(real_corr - gen_corr)
        
        metrics = {
            'corr_frobenius_norm': np.linalg.norm(corr_diff, 'fro'),
            'corr_max_diff': np.max(corr_diff),
            'corr_mean_diff': np.mean(corr_diff),
        }
        
        return metrics
    
    def compute_privacy_metrics(
        self,
        real_data: np.ndarray,
        generated_data: np.ndarray,
        k: int = 5
    ) -> Dict[str, float]:
        """
        Compute privacy metrics (distance to closest real sample).
        
        Args:
            real_data: Real flow data
            generated_data: Generated flow data
            k: Number of nearest neighbors to consider
            
        Returns:
            Dictionary of privacy metrics
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Normalize data for distance computation
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        real_norm = scaler.fit_transform(real_data)
        gen_norm = scaler.transform(generated_data)
        
        # Find k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean')
        nbrs.fit(real_norm)
        
        distances, indices = nbrs.kneighbors(gen_norm)
        
        metrics = {
            'min_distance_to_real': np.min(distances[:, 0]),
            'mean_distance_to_real': np.mean(distances[:, 0]),
            'std_distance_to_real': np.std(distances[:, 0]),
        }
        
        return metrics
    
    def evaluate(
        self,
        real_data: np.ndarray,
        generated_data: np.ndarray,
        output_dir: str = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of generated flow statistics.
        
        Args:
            real_data: Real flow data
            generated_data: Generated flow data
            output_dir: Directory to save plots (optional)
            
        Returns:
            Dictionary of all metrics
        """
        results = {}
        
        # Distribution metrics
        dist_metrics = self.compute_distribution_metrics(real_data, generated_data)
        results.update(dist_metrics)
        
        # Correlation metrics
        corr_metrics = self.compute_correlation_metrics(real_data, generated_data)
        results.update(corr_metrics)
        
        # Privacy metrics
        privacy_metrics = self.compute_privacy_metrics(real_data, generated_data)
        results.update(privacy_metrics)
        
        # Create visualizations if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.create_visualizations(real_data, generated_data, output_dir)
        
        return results
    
    def create_visualizations(
        self,
        real_data: np.ndarray,
        generated_data: np.ndarray,
        output_dir: str
    ):
        """Create visualization plots."""
        
        # 1. Feature distributions
        n_features = len(self.feature_names)
        fig, axes = plt.subplots(2, (n_features + 1) // 2, figsize=(15, 8))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, feature_name in enumerate(self.feature_names):
            ax = axes[i]
            ax.hist(real_data[:, i], bins=50, alpha=0.7, label='Real', density=True)
            ax.hist(generated_data[:, i], bins=50, alpha=0.7, label='Generated', density=True)
            ax.set_title(f'{feature_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove extra subplots
        for i in range(n_features, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=300)
        plt.close()
        
        # 2. Correlation matrices
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        real_corr = np.corrcoef(real_data.T)
        gen_corr = np.corrcoef(generated_data.T)
        
        im1 = ax1.imshow(real_corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax1.set_title('Real Data Correlation')
        ax1.set_xticks(range(len(self.feature_names)))
        ax1.set_yticks(range(len(self.feature_names)))
        ax1.set_xticklabels(self.feature_names, rotation=45)
        ax1.set_yticklabels(self.feature_names)
        
        im2 = ax2.imshow(gen_corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax2.set_title('Generated Data Correlation')
        ax2.set_xticks(range(len(self.feature_names)))
        ax2.set_yticks(range(len(self.feature_names)))
        ax2.set_xticklabels(self.feature_names, rotation=45)
        ax2.set_yticklabels(self.feature_names)
        
        diff_corr = np.abs(real_corr - gen_corr)
        im3 = ax3.imshow(diff_corr, cmap='Reds', vmin=0, vmax=1)
        ax3.set_title('Correlation Difference')
        ax3.set_xticks(range(len(self.feature_names)))
        ax3.set_yticks(range(len(self.feature_names)))
        ax3.set_xticklabels(self.feature_names, rotation=45)
        ax3.set_yticklabels(self.feature_names)
        
        plt.colorbar(im1, ax=ax1)
        plt.colorbar(im2, ax=ax2)
        plt.colorbar(im3, ax=ax3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrices.png'), dpi=300)
        plt.close()
        
        # 3. 2D visualization using PCA
        if real_data.shape[1] > 2:
            pca = PCA(n_components=2)
            real_pca = pca.fit_transform(real_data)
            gen_pca = pca.transform(generated_data)
            
            plt.figure(figsize=(10, 6))
            plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.6, label='Real', s=20)
            plt.scatter(gen_pca[:, 0], gen_pca[:, 1], alpha=0.6, label='Generated', s=20)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('PCA Visualization')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'pca_visualization.png'), dpi=300)
            plt.close()
        
        # 4. Summary statistics table
        real_stats = pd.DataFrame(real_data, columns=self.feature_names).describe()
        gen_stats = pd.DataFrame(generated_data, columns=self.feature_names).describe()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Real data statistics
        sns.heatmap(real_stats, annot=True, fmt='.2f', cmap='Blues', ax=ax1)
        ax1.set_title('Real Data Statistics')
        
        # Generated data statistics
        sns.heatmap(gen_stats, annot=True, fmt='.2f', cmap='Oranges', ax=ax2)
        ax2.set_title('Generated Data Statistics')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'summary_statistics.png'), dpi=300)
        plt.close()


def generate_flows(
    model: FlowBD3LM,
    tokenizer: FlowTokenizer,
    n_samples: int = 1000,
    block_size: int = 1,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: torch.device = None
) -> np.ndarray:
    """
    Generate flow statistics using the trained model.
    
    Args:
        model: Trained FlowBD3LM model
        tokenizer: Fitted tokenizer
        n_samples: Number of samples to generate
        block_size: Block size for generation
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        device: Device to run on
        
    Returns:
        Generated flow statistics array
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    model.to(device)
    
    generated_samples = []
    batch_size = min(64, n_samples)  # Generate in batches to avoid memory issues
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - i)
            
            # Start with mask tokens
            input_ids = torch.full(
                (current_batch_size, len(tokenizer.feature_names)),
                fill_value=tokenizer.mask_token_id,
                dtype=torch.long,
                device=device
            )
            
            # Use iterative demasking instead of the generate method
            # This is more appropriate for masked diffusion models
            generated = iterative_demask(
                model=model,
                input_ids=input_ids,
                mask_token_id=tokenizer.mask_token_id,
                actual_n_bins=tokenizer.actual_n_bins,
                vocab_size=tokenizer.vocab_size,
                n_steps=20,  # Number of demasking steps
                temperature=temperature,
                top_p=top_p,
                device=device
            )
            
            # Convert to numpy and detokenize
            generated_np = tokenizer.detokenize(generated.cpu())
            generated_samples.append(generated_np)
    
    return np.concatenate(generated_samples, axis=0)


def iterative_demask(
    model: FlowBD3LM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    actual_n_bins: dict,
    vocab_size: int,
    n_steps: int = 20,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: torch.device = None
) -> torch.Tensor:
    """
    Iteratively demask tokens using the trained model.
    
    Args:
        model: Trained model
        input_ids: Initial input with mask tokens
        mask_token_id: ID of mask token
        actual_n_bins: Dictionary of actual bins per feature
        vocab_size: Total vocabulary size
        n_steps: Number of demasking steps
        temperature: Sampling temperature
        top_p: Top-p parameter
        device: Device to run on
        
    Returns:
        Generated tokens
    """
    batch_size, seq_len = input_ids.shape
    generated = input_ids.clone()
    feature_names = list(actual_n_bins.keys())
    
    for step in range(n_steps):
        # Count remaining mask tokens
        mask_positions = (generated == mask_token_id)
        n_masked = mask_positions.sum().item()
        
        if n_masked == 0:
            break
            
        # Forward pass
        outputs = model(generated, block_size=1)
        logits = outputs['logits']
        
        # Apply temperature
        logits = logits / temperature
        
        # Create feature-specific valid token masks
        for pos_idx in range(seq_len):
            if pos_idx < len(feature_names):
                feature_name = feature_names[pos_idx]
                max_valid_token = actual_n_bins[feature_name]
                
                # Mask out invalid tokens for this feature
                logits[:, pos_idx, max_valid_token:] = -float('inf')
        
        # Apply top-p sampling
        if top_p < 1.0:
            # Apply top-p to each position
            for b in range(batch_size):
                for s in range(seq_len):
                    if mask_positions[b, s]:
                        sorted_logits, sorted_indices = torch.sort(logits[b, s], descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                        sorted_indices_to_remove[0] = False
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                        logits[b, s][indices_to_remove] = -float('inf')
        
        # Sample tokens for masked positions
        probs = F.softmax(logits, dim=-1)
        
        # Decide how many tokens to unmask in this step
        # Use a schedule: unmask more tokens in later steps
        unmask_ratio = (step + 1) / n_steps
        n_to_unmask = max(1, int(unmask_ratio * n_masked))
        
        # Sample new tokens only for mask positions
        for b in range(batch_size):
            masked_pos = torch.where(mask_positions[b])[0]
            if len(masked_pos) == 0:
                continue
                
            # Sample tokens for masked positions
            sampled_tokens = torch.multinomial(probs[b, masked_pos], num_samples=1).squeeze(-1)
            
            # Choose which positions to unmask (randomly)
            if len(masked_pos) > n_to_unmask:
                positions_to_unmask = torch.randperm(len(masked_pos))[:n_to_unmask]
                masked_pos = masked_pos[positions_to_unmask]
                sampled_tokens = sampled_tokens[positions_to_unmask]
            
            # Update the generated sequence
            generated[b, masked_pos] = sampled_tokens
    
    # Fill any remaining masked tokens with random valid tokens per feature
    remaining_masks = (generated == mask_token_id)
    if remaining_masks.any():
        for b in range(batch_size):
            for pos in range(seq_len):
                if remaining_masks[b, pos] and pos < len(feature_names):
                    feature_name = feature_names[pos]
                    max_valid_token = actual_n_bins[feature_name]
                    generated[b, pos] = torch.randint(0, max_valid_token, (1,), device=device)[0]
    
    return generated


def evaluate_model(
    model_path: str,
    tokenizer_path: str,
    test_data: np.ndarray,
    feature_names: List[str],
    output_dir: str,
    n_generated: int = 1000,
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model_path: Path to saved model checkpoint
        tokenizer_path: Path to saved tokenizer
        test_data: Test data for comparison
        feature_names: Feature names
        output_dir: Output directory for results
        n_generated: Number of samples to generate
        
    Returns:
        Evaluation results dictionary
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    tokenizer = FlowTokenizer.load(tokenizer_path)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
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
    
    # Generate samples
    print(f"Generating {n_generated} flow samples...")
    generated_data = generate_flows(
        model=model,
        tokenizer=tokenizer,
        n_samples=n_generated,
        block_size=config['block_size'],
        device=device
    )
    
    # Evaluate
    evaluator = FlowEvaluator(feature_names)
    results = evaluator.evaluate(
        real_data=test_data,
        generated_data=generated_data,
        output_dir=output_dir
    )
    
    # Save results
    import json
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save generated data
    pd.DataFrame(generated_data, columns=feature_names).to_csv(
        os.path.join(output_dir, 'generated_flows.csv'), index=False
    )
    
    return results