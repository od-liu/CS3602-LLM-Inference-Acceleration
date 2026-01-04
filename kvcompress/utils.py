"""
Utility Functions for KV Cache Compression

This module provides common utility functions used across compression methods.
"""

from typing import List, Tuple, Union
import torch
from transformers import DynamicCache


def to_dynamic_cache(
    past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]
) -> DynamicCache:
    """
    Convert list of (key, value) tuples to DynamicCache object.
    
    Args:
        past_key_values: List of (key, value) tuples
    
    Returns:
        DynamicCache object
    """
    cache = DynamicCache()
    for layer_idx, (keys, values) in enumerate(past_key_values):
        cache.update(keys, values, layer_idx)
    return cache


def normalize_kv_cache(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache]
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Convert KV cache to list format.
    
    Args:
        past_key_values: KV cache in any supported format
    
    Returns:
        KV cache as list of (key, value) tuples
    """
    if hasattr(past_key_values, 'to_legacy_cache'):
        return past_key_values.to_legacy_cache()
    return list(past_key_values)


def get_cache_size_mb(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache]
) -> float:
    """
    Calculate KV cache size in megabytes.
    
    Args:
        past_key_values: KV cache
    
    Returns:
        Size in megabytes
    """
    past_key_values = normalize_kv_cache(past_key_values)
    
    total_size = 0
    for keys, values in past_key_values:
        total_size += keys.element_size() * keys.nelement()
        total_size += values.element_size() * values.nelement()
    return total_size / (1024 ** 2)


def get_cache_info(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache]
) -> dict:
    """
    Get detailed information about KV cache.
    
    Args:
        past_key_values: KV cache
    
    Returns:
        Dict with cache information
    """
    past_key_values = normalize_kv_cache(past_key_values)
    
    if not past_key_values:
        return {"num_layers": 0, "seq_lengths": [], "total_size_mb": 0}
    
    seq_lengths = [keys.size(2) for keys, values in past_key_values]
    
    return {
        "num_layers": len(past_key_values),
        "seq_lengths": seq_lengths,
        "min_seq_len": min(seq_lengths),
        "max_seq_len": max(seq_lengths),
        "avg_seq_len": sum(seq_lengths) / len(seq_lengths),
        "total_size_mb": get_cache_size_mb(past_key_values)
    }


def get_seq_len(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache],
    layer_idx: int = 0
) -> int:
    """
    Get sequence length from KV cache.
    
    Args:
        past_key_values: KV cache
        layer_idx: Layer index to check (default: 0)
    
    Returns:
        Sequence length
    """
    past_key_values = normalize_kv_cache(past_key_values)
    
    if not past_key_values or layer_idx >= len(past_key_values):
        return 0
    
    return past_key_values[layer_idx][0].size(2)

