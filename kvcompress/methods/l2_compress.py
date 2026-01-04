"""
L2 Norm-Based KV Cache Compression (KnormPress)

This module implements the original KnormPress algorithm that compresses
KV cache by keeping tokens with the lowest L2 norms.

Reference: "A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression"
"""

from typing import List, Tuple, Union
from math import ceil
import torch
from transformers import DynamicCache

from ..utils import normalize_kv_cache


def l2_compress(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache],
    keep_ratio: float = 1.0,
    prune_after: int = 1000,
    skip_layers: List[int] = [0, 1],
    **kwargs
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compress KV cache by keeping tokens with the lowest L2 norms.
    
    This is the original KnormPress algorithm that compresses by ratio.
    Tokens with lower L2 norms correlate with higher attention scores,
    so keeping them preserves the most important information.
    
    Args:
        past_key_values: KV cache, shape (batch, heads, seq_len, head_dim)
        keep_ratio: Fraction of tokens to keep (0.0 to 1.0)
        prune_after: Only compress if sequence length > this value
        skip_layers: Layer indices to skip compression (typically first layers)
        **kwargs: Additional arguments (ignored, for interface compatibility)
    
    Returns:
        Compressed KV cache as list of (key, value) tuples
    
    Example:
        >>> compressed = l2_compress(past_key_values, keep_ratio=0.8)
    """
    # Convert to list format if needed
    past_key_values = list(normalize_kv_cache(past_key_values))
    
    if keep_ratio >= 1.0:
        return past_key_values
    
    for layer_idx, (keys, values) in enumerate(past_key_values):
        seq_len = keys.size(2)
        
        # Skip if sequence is short
        if seq_len <= prune_after:
            continue
        
        # Skip specified layers
        if layer_idx in skip_layers:
            continue
        
        tokens_to_keep = ceil(keep_ratio * seq_len)
        
        if tokens_to_keep >= seq_len:
            continue
        
        batch_size, num_heads, seq_len, head_dim = keys.shape
        
        # Compute L2 norm for each token
        token_norms = torch.norm(keys, p=2, dim=-1)
        
        # Sort by norm (ascending = low norm = important tokens)
        sorted_indices = token_norms.argsort(dim=-1)
        
        # Select top tokens (lowest norms)
        indices_to_keep = sorted_indices[:, :, :tokens_to_keep]
        
        # Maintain temporal order
        indices_to_keep_sorted, _ = torch.sort(indices_to_keep, dim=-1)
        
        # Expand indices for gather operation
        indices_expanded = indices_to_keep_sorted.unsqueeze(-1).expand(
            batch_size, num_heads, tokens_to_keep, head_dim
        )
        
        # Gather selected tokens
        compressed_keys = torch.gather(keys, dim=2, index=indices_expanded)
        compressed_values = torch.gather(values, dim=2, index=indices_expanded)
        
        past_key_values[layer_idx] = (compressed_keys, compressed_values)
    
    return past_key_values


__all__ = ['l2_compress']

