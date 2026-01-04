"""
Base classes and interfaces for KV Cache Compression Methods

This module defines the common interface that all compression methods should follow.
"""

from typing import List, Tuple, Union, Protocol, runtime_checkable
import torch
from transformers import DynamicCache


@runtime_checkable
class CompressFn(Protocol):
    """Protocol defining the interface for compression functions."""
    
    def __call__(
        self,
        past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache],
        **kwargs
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compress KV cache.
        
        Args:
            past_key_values: KV cache, either as list of (key, value) tuples
                           or as DynamicCache object.
                           Shape of each tensor: (batch, heads, seq_len, head_dim)
            **kwargs: Method-specific parameters
        
        Returns:
            Compressed KV cache as list of (key, value) tuples
        """
        ...


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

