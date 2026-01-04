"""
Fixed-Size L2 Norm-Based KV Cache Compression

This module implements fixed-size KV cache compression with multiple
eviction strategies based on L2 norms.
"""

from typing import List, Tuple, Union, Literal
import torch
from transformers import DynamicCache

from ..utils import normalize_kv_cache


def fix_size_l2_compress(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache],
    fix_kv_size: int = 1024,
    keep_ratio: float = 0.0,
    strategy: Literal["keep_low", "keep_high", "random"] = "keep_low",
    skip_layers: List[int] = [0, 1],
    **kwargs
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Fixed-size KV cache compression with eviction strategies.
    
    This method maintains a fixed maximum KV cache size by evicting tokens
    when the cache exceeds fix_kv_size. Recent tokens (controlled by keep_ratio)
    are never evicted; eviction only happens in the older portion.
    
    Algorithm:
    1. If cache size <= fix_kv_size, no eviction needed
    2. Calculate protected_length = fix_kv_size * keep_ratio (recent tokens to keep)
    3. Eviction zone = tokens[0 : seq_len - protected_length]
    4. From eviction zone, keep (fix_kv_size - protected_length) tokens based on strategy
    5. Combine kept eviction zone tokens + protected recent tokens
    
    Args:
        past_key_values: KV cache, shape (batch, heads, seq_len, head_dim)
        fix_kv_size: Maximum number of tokens to keep in cache
        keep_ratio: Fraction of fix_kv_size to protect (most recent tokens)
                   Default 0.0 means all tokens can be evicted based on strategy
                   Example: keep_ratio=0.2, fix_kv_size=1000 -> last 200 tokens protected
        strategy: Eviction strategy for non-protected tokens
                  - "keep_low": Keep tokens with low L2 norm (important tokens)
                  - "keep_high": Keep tokens with high L2 norm
                  - "random": Random eviction
        skip_layers: Layer indices to skip compression
        **kwargs: Additional arguments (ignored, for interface compatibility)
    
    Returns:
        Compressed KV cache with at most fix_kv_size tokens per layer
    
    Example:
        >>> # Keep max 512 tokens, protect last 20% (102 tokens)
        >>> compressed = fix_size_l2_compress(
        ...     past_key_values,
        ...     fix_kv_size=512,
        ...     keep_ratio=0.2,
        ...     strategy="keep_low"
        ... )
    """
    # Convert to list format if needed
    past_key_values = list(normalize_kv_cache(past_key_values))
    
    for layer_idx, (keys, values) in enumerate(past_key_values):
        seq_len = keys.size(2)
        
        # No eviction needed if within limit
        if seq_len <= fix_kv_size:
            continue
        
        # Skip specified layers
        if layer_idx in skip_layers:
            continue
        
        batch_size, num_heads, seq_len, head_dim = keys.shape
        
        # Calculate protected zone (recent tokens that won't be evicted)
        protected_length = int(fix_kv_size * keep_ratio)
        protected_length = min(protected_length, seq_len)
        
        # Eviction zone: everything except the protected recent tokens
        eviction_zone_end = seq_len - protected_length
        
        # How many tokens to keep from eviction zone
        tokens_to_keep_from_eviction = fix_kv_size - protected_length
        
        if tokens_to_keep_from_eviction <= 0:
            # Only keep protected recent tokens
            final_keys = keys[:, :, -protected_length:, :]
            final_values = values[:, :, -protected_length:, :]
            past_key_values[layer_idx] = (final_keys, final_values)
            continue
        
        if eviction_zone_end <= tokens_to_keep_from_eviction:
            # No need to evict, keep all from eviction zone
            continue
        
        # Get eviction zone keys for computing norms
        eviction_keys = keys[:, :, :eviction_zone_end, :]
        eviction_values = values[:, :, :eviction_zone_end, :]
        
        # Select tokens to keep from eviction zone based on strategy
        if strategy == "keep_low":
            # Keep tokens with lowest L2 norm (most important)
            token_norms = torch.norm(eviction_keys, p=2, dim=-1)
            sorted_indices = token_norms.argsort(dim=-1)
            indices_to_keep = sorted_indices[:, :, :tokens_to_keep_from_eviction]
            
        elif strategy == "keep_high":
            # Keep tokens with highest L2 norm
            token_norms = torch.norm(eviction_keys, p=2, dim=-1)
            sorted_indices = token_norms.argsort(dim=-1, descending=True)
            indices_to_keep = sorted_indices[:, :, :tokens_to_keep_from_eviction]
            
        elif strategy == "random":
            # Random selection
            indices_to_keep = torch.stack([
                torch.stack([
                    torch.randperm(eviction_zone_end, device=keys.device)[:tokens_to_keep_from_eviction]
                    for _ in range(num_heads)
                ])
                for _ in range(batch_size)
            ])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # CRITICAL: Sort indices to maintain temporal order
        indices_to_keep_sorted, _ = torch.sort(indices_to_keep, dim=-1)
        
        # Expand indices for gather
        indices_expanded = indices_to_keep_sorted.unsqueeze(-1).expand(
            batch_size, num_heads, tokens_to_keep_from_eviction, head_dim
        )
        
        # Gather selected tokens from eviction zone
        kept_eviction_keys = torch.gather(eviction_keys, dim=2, index=indices_expanded)
        kept_eviction_values = torch.gather(eviction_values, dim=2, index=indices_expanded)
        
        # Get protected recent tokens
        if protected_length > 0:
            protected_keys = keys[:, :, -protected_length:, :]
            protected_values = values[:, :, -protected_length:, :]
            
            # Concatenate: kept eviction tokens + protected recent tokens
            final_keys = torch.cat([kept_eviction_keys, protected_keys], dim=2)
            final_values = torch.cat([kept_eviction_values, protected_values], dim=2)
        else:
            final_keys = kept_eviction_keys
            final_values = kept_eviction_values
        
        past_key_values[layer_idx] = (final_keys, final_values)
    
    return past_key_values


__all__ = ['fix_size_l2_compress']

