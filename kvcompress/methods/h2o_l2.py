"""
H2O-L2: Heavy-Hitter Oracle with L2 Norm Approximation

This module implements an H2O-inspired KV cache compression method that uses
L2 norms as a proxy for attention scores. The original H2O algorithm requires
accumulated attention scores, but research shows low L2 norm correlates with
high attention, making this a practical approximation.

The method combines three strategies:
1. Attention Sinks: Keep initial tokens (like StreamingLLM)
2. Heavy Hitters: Keep tokens with lowest L2 norms from middle section
3. Recent Window: Keep most recent tokens

Reference: H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs
           https://arxiv.org/abs/2306.14048
"""

from typing import List, Tuple, Union
import torch
from transformers import DynamicCache

from ..utils import normalize_kv_cache


def h2o_l2_compress(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache],
    start_size: int = 4,
    heavy_hitter_size: int = 64,
    recent_size: int = 444,
    skip_layers: List[int] = [],
    **kwargs
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    H2O-L2 compression: attention sinks + heavy hitters (L2-based) + recent tokens.
    
    This method approximates the H2O algorithm by using L2 norms instead of
    attention scores. Tokens with lower L2 norms tend to receive higher attention,
    so we keep those as "heavy hitters".
    
    Cache structure:
        [attention sinks (0:start_size)] + 
        [heavy hitters from middle (selected by lowest L2)] + 
        [recent tokens (seq_len-recent_size:seq_len)]
    
    Args:
        past_key_values: KV cache, shape (batch, heads, seq_len, head_dim)
        start_size: Number of initial tokens to keep as attention sinks (default: 4)
        heavy_hitter_size: Number of heavy hitter tokens to keep from middle (default: 64)
        recent_size: Number of recent tokens to keep (default: 444)
                    Total cache = start_size + heavy_hitter_size + recent_size = 512
        skip_layers: Layer indices to skip compression (default: empty)
        **kwargs: Additional arguments (ignored, for interface compatibility)
    
    Returns:
        Compressed KV cache with at most (start_size + heavy_hitter_size + recent_size) tokens
    
    Example:
        >>> # Default: 4 sinks + 64 heavy hitters + 444 recent = 512 total
        >>> compressed = h2o_l2_compress(past_key_values)
        
        >>> # Custom configuration
        >>> compressed = h2o_l2_compress(
        ...     past_key_values,
        ...     start_size=4,
        ...     heavy_hitter_size=128,
        ...     recent_size=380  # 4 + 128 + 380 = 512
        ... )
    """
    # Convert to list format if needed
    past_key_values = list(normalize_kv_cache(past_key_values))
    
    if not past_key_values:
        return past_key_values
    
    total_cache_size = start_size + heavy_hitter_size + recent_size
    
    for layer_idx, (keys, values) in enumerate(past_key_values):
        seq_len = keys.size(2)
        
        # No compression needed if within cache size
        if seq_len <= total_cache_size:
            continue
        
        # Skip specified layers
        if layer_idx in skip_layers:
            continue
        
        batch_size, num_heads, _, head_dim = keys.shape
        
        # Define regions
        # Region 1: Attention sinks [0, start_size)
        # Region 2: Middle tokens [start_size, seq_len - recent_size) - candidates for heavy hitters
        # Region 3: Recent tokens [seq_len - recent_size, seq_len)
        
        middle_start = start_size
        middle_end = seq_len - recent_size
        
        # If middle region is too small, fall back to StreamingLLM behavior
        if middle_end <= middle_start:
            # Just keep attention sinks + recent
            sink_keys = keys[:, :, :start_size, :]
            sink_values = values[:, :, :start_size, :]
            recent_keys = keys[:, :, -recent_size:, :]
            recent_values = values[:, :, -recent_size:, :]
            
            compressed_keys = torch.cat([sink_keys, recent_keys], dim=2)
            compressed_values = torch.cat([sink_values, recent_values], dim=2)
            past_key_values[layer_idx] = (compressed_keys, compressed_values)
            continue
        
        # Get attention sinks
        sink_keys = keys[:, :, :start_size, :]
        sink_values = values[:, :, :start_size, :]
        
        # Get middle region for heavy hitter selection
        middle_keys = keys[:, :, middle_start:middle_end, :]
        middle_values = values[:, :, middle_start:middle_end, :]
        middle_len = middle_keys.size(2)
        
        # Calculate L2 norms for middle tokens
        # Lower L2 norm = higher importance (correlates with attention)
        token_norms = torch.norm(middle_keys, p=2, dim=-1)  # (batch, heads, middle_len)
        
        # Select heavy hitters (tokens with lowest L2 norms)
        num_to_keep = min(heavy_hitter_size, middle_len)
        
        # Sort by norm (ascending = low norm = important)
        sorted_indices = token_norms.argsort(dim=-1)
        indices_to_keep = sorted_indices[:, :, :num_to_keep]
        
        # Maintain temporal order for selected tokens
        indices_to_keep_sorted, _ = torch.sort(indices_to_keep, dim=-1)
        
        # Expand indices for gather
        indices_expanded = indices_to_keep_sorted.unsqueeze(-1).expand(
            batch_size, num_heads, num_to_keep, head_dim
        )
        
        # Gather heavy hitter tokens
        heavy_keys = torch.gather(middle_keys, dim=2, index=indices_expanded)
        heavy_values = torch.gather(middle_values, dim=2, index=indices_expanded)
        
        # Get recent tokens
        recent_keys = keys[:, :, -recent_size:, :]
        recent_values = values[:, :, -recent_size:, :]
        
        # Concatenate: sinks + heavy hitters + recent
        compressed_keys = torch.cat([sink_keys, heavy_keys, recent_keys], dim=2)
        compressed_values = torch.cat([sink_values, heavy_values, recent_values], dim=2)
        
        past_key_values[layer_idx] = (compressed_keys, compressed_values)
    
    return past_key_values


__all__ = ['h2o_l2_compress']

