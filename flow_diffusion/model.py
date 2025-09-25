"""
Block Diffusion Language Model adapted for network flow statistics.
Based on the BD3-LM architecture from the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
import numpy as np


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings."""
    
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Create positional encoding
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate rotary positional embeddings."""
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to query and key tensors."""
    cos = cos.unsqueeze(1)  # [seq_len, 1, dim]
    sin = sin.unsqueeze(1)  # [seq_len, 1, dim]
    
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-head attention with rotary positional embeddings."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary positional embeddings
        rotary_emb = self.rotary_emb(seq_len, x.device)
        cos = torch.cos(rotary_emb).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(rotary_emb).unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward layers."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out = self.attention(self.norm1(x), mask)
        x = x + attn_out
        
        # Feed-forward with residual connection
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        
        return x


class FlowBD3LM(nn.Module):
    """Block Diffusion Model for Flow Statistics."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        mask_token_id: int = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mask_token_id = mask_token_id or (vocab_size - 1)
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def create_block_causal_mask(self, seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
        """Create block-causal attention mask."""
        mask = torch.zeros(seq_len, seq_len, device=device)
        
        for i in range(0, seq_len, block_size):
            block_end = min(i + block_size, seq_len)
            # Allow attention within block and to previous blocks
            mask[i:block_end, :block_end] = 1
            
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        
    def forward(
        self,
        input_ids: torch.Tensor,
        block_size: int = 1,
        return_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            block_size: Size of blocks for block-causal attention
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Dictionary with logits and optionally hidden states
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Create block-causal mask
        mask = self.create_block_causal_mask(seq_len, block_size, device)
        
        # Apply transformer blocks
        hidden_states = []
        for block in self.blocks:
            x = block(x, mask)
            if return_hidden_states:
                hidden_states.append(x)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        outputs = {'logits': logits}
        if return_hidden_states:
            outputs['hidden_states'] = hidden_states
            
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        block_size: int = 1,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate new tokens using the model.
        
        Args:
            input_ids: Initial token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            block_size: Block size for generation
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if generated.shape[1] >= self.max_seq_len:
                    break
                    
                # Forward pass
                outputs = self.forward(generated, block_size=block_size)
                logits = outputs['logits'][:, -1, :]  # Last token logits
                
                # Apply temperature
                logits = logits / temperature
                
                # Top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated


def create_noise_schedule(t: torch.Tensor, schedule_type: str = 'linear') -> torch.Tensor:
    """Create noise schedule for diffusion training."""
    if schedule_type == 'linear':
        alpha_t = 1.0 - t
    elif schedule_type == 'cosine':
        alpha_t = torch.cos(0.5 * np.pi * t)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    return alpha_t


def compute_diffusion_loss(
    model: FlowBD3LM,
    input_ids: torch.Tensor,
    block_size: int = 1,
    mask_prob: float = 0.15,
) -> torch.Tensor:
    """
    Compute block diffusion loss for training.
    
    Args:
        model: The BD3-LM model
        input_ids: Clean token IDs [batch_size, seq_len]
        block_size: Block size for diffusion
        mask_prob: Probability of masking tokens
        
    Returns:
        Loss tensor
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    # Sample time steps
    t = torch.rand(batch_size, device=device)
    alpha_t = create_noise_schedule(t)
    
    # Create noisy inputs by randomly masking tokens
    noisy_input_ids = input_ids.clone()
    targets = input_ids.clone()
    
    # Sample masking probabilities based on time step
    mask_probs = (1.0 - alpha_t).unsqueeze(1).expand(-1, seq_len)
    mask_matrix = torch.rand(batch_size, seq_len, device=device) < mask_probs
    
    # Apply masking
    noisy_input_ids[mask_matrix] = model.mask_token_id
    
    # Forward pass
    outputs = model(noisy_input_ids, block_size=block_size)
    logits = outputs['logits']
    
    # Compute loss only on masked tokens
    loss_mask = mask_matrix
    
    # Reshape for cross-entropy loss
    logits_flat = logits.view(-1, model.vocab_size)
    targets_flat = targets.view(-1)
    loss_mask_flat = loss_mask.view(-1)
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
    
    # Apply mask and compute weighted average
    masked_loss = loss * loss_mask_flat.float()
    
    # Weight by inverse alpha_t for diffusion objective
    alpha_t_expanded = alpha_t.unsqueeze(1).expand(-1, seq_len).reshape(-1)
    weights = (1.0 - alpha_t_expanded) / torch.clamp(alpha_t_expanded, min=1e-8)
    weighted_loss = masked_loss * weights
    
    return weighted_loss.sum() / loss_mask_flat.sum().clamp(min=1)