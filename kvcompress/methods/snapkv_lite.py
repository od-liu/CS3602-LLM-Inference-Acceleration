"""
SnapKV-Lite: Simplified SnapKV with Observation Window Voting

This module implements a lightweight version of SnapKV that uses an observation
window to identify important tokens through voting mechanism. Instead of using
actual attention patterns, we use L2 norms as importance proxies.

The key insight from SnapKV is that tokens receiving consistent attention
across recent queries are more important. We approximate this by looking at
which tokens have consistently low L2 norms (high importance) when evaluated
against a sliding observation window.

Reference: SnapKV: LLM Knows What You are Looking for Before Generation
           https://arxiv.org/abs/2404.14469
"""

from typing import List, Tuple, Union
import torch
from transformers import DynamicCache

from ..utils import normalize_kv_cache


def snapkv_lite_compress(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache],
    observation_window: int = 32,
    keep_size: int = 512,
    pooling_kernel: int = 5,
    skip_layers: List[int] = [],
    **kwargs
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    SnapKV-Lite compression: observation window voting with L2 norm importance.
    
    This method uses an observation window to vote on token importance:
    1. Use last `observation_window` tokens as the observation context
    2. Calculate importance scores for prefix tokens based on L2 norm patterns
    3. Apply pooling to smooth importance scores
    4. Keep top-k most important tokens + observation window
    
    Args:
        past_key_values: KV cache, shape (batch, heads, seq_len, head_dim)
        observation_window: Number of recent tokens used for voting (default: 32)
        keep_size: Total number of tokens to keep including observation window (default: 512)
        pooling_kernel: Kernel size for importance smoothing (default: 5)
        skip_layers: Layer indices to skip compression (default: empty)
        **kwargs: Additional arguments (ignored, for interface compatibility)
    
    Returns:
        Compressed KV cache with at most keep_size tokens
    
    Example:
        >>> # Keep 512 tokens total with 32-token observation window
        >>> compressed = snapkv_lite_compress(
        ...     past_key_values,
        ...     observation_window=32,
        ...     keep_size=512
        ... )
    """
    # Convert to list format if needed
    past_key_values = list(normalize_kv_cache(past_key_values))
    
    if not past_key_values:
        return past_key_values
    
    for layer_idx, (keys, values) in enumerate(past_key_values):
        seq_len = keys.size(2)
        
        # No compression needed if within limit
        if seq_len <= keep_size:
            continue
        
        # Skip specified layers
        if layer_idx in skip_layers:
            continue
        
        batch_size, num_heads, _, head_dim = keys.shape
        
        # Separate prefix and observation window
        # Prefix: tokens to select from [0, seq_len - observation_window)
        # Observation: recent tokens that are always kept [seq_len - observation_window, seq_len)
        
        prefix_len = seq_len - observation_window
        if prefix_len <= 0:
            # Not enough tokens for prefix, keep everything
            continue
        
        prefix_keys = keys[:, :, :prefix_len, :]
        prefix_values = values[:, :, :prefix_len, :]
        obs_keys = keys[:, :, -observation_window:, :]
        obs_values = values[:, :, -observation_window:, :]
        
        # Calculate importance scores for prefix tokens
        # Use L2 norm as importance proxy (lower = more important)
        # We invert it so higher score = more important
        prefix_norms = torch.norm(prefix_keys, p=2, dim=-1)  # (batch, heads, prefix_len)
        
        # Invert: lower norm -> higher importance
        max_norm = prefix_norms.max(dim=-1, keepdim=True)[0] + 1e-6
        importance_scores = max_norm - prefix_norms  # Higher = more important
        
        # Apply local pooling to smooth importance scores
        # This helps capture token importance patterns, similar to SnapKV's voting
        if pooling_kernel > 1 and prefix_len >= pooling_kernel:
            # Reshape for 1D convolution: (batch * heads, 1, prefix_len)
            scores_flat = importance_scores.view(batch_size * num_heads, 1, prefix_len)
            
            # Average pooling
            padding = pooling_kernel // 2
            pooled = torch.nn.functional.avg_pool1d(
                scores_flat, 
                kernel_size=pooling_kernel, 
                stride=1, 
                padding=padding
            )
            
            # Handle edge cases where pooling changes length
            if pooled.size(-1) != prefix_len:
                pooled = pooled[:, :, :prefix_len]
            
            importance_scores = pooled.view(batch_size, num_heads, -1)
        
        # Select top-k tokens from prefix
        # keep_size - observation_window = number of prefix tokens to keep
        num_prefix_to_keep = keep_size - observation_window
        num_prefix_to_keep = min(num_prefix_to_keep, prefix_len)
        
        if num_prefix_to_keep <= 0:
            # Just keep observation window
            past_key_values[layer_idx] = (obs_keys, obs_values)
            continue
        
        # Get top-k indices (highest importance scores)
        _, top_indices = torch.topk(importance_scores, num_prefix_to_keep, dim=-1)
        
        # Sort indices to maintain temporal order
        top_indices_sorted, _ = torch.sort(top_indices, dim=-1)
        
        # Expand indices for gather
        indices_expanded = top_indices_sorted.unsqueeze(-1).expand(
            batch_size, num_heads, num_prefix_to_keep, head_dim
        )
        
        # Gather selected prefix tokens
        selected_prefix_keys = torch.gather(prefix_keys, dim=2, index=indices_expanded)
        selected_prefix_values = torch.gather(prefix_values, dim=2, index=indices_expanded)
        
        # Concatenate: selected prefix + observation window
        compressed_keys = torch.cat([selected_prefix_keys, obs_keys], dim=2)
        compressed_values = torch.cat([selected_prefix_values, obs_values], dim=2)
        
        past_key_values[layer_idx] = (compressed_keys, compressed_values)
    
    return past_key_values


__all__ = ['snapkv_lite_compress']

