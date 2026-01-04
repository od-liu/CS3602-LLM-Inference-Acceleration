"""
StreamingLLM KV Cache Compression

This module implements the StreamingLLM approach for KV cache compression,
which maintains attention sinks (initial tokens) alongside a sliding window
of recent tokens.

Reference: "Efficient Streaming Language Models with Attention Sinks" (ICLR 2024)
           https://github.com/mit-han-lab/streaming-llm
"""

from typing import List, Tuple, Union
import torch
from transformers import DynamicCache

from ..utils import normalize_kv_cache


def streaming_llm_compress(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache],
    start_size: int = 4,
    recent_size: int = 508,
    skip_layers: List[int] = [],
    **kwargs
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    StreamingLLM compression: keep attention sinks + recent tokens.
    
    This method implements the core insight from the StreamingLLM paper:
    LLMs allocate significant attention to initial tokens ("attention sinks")
    regardless of their semantic importance. By preserving these initial tokens
    alongside recent tokens, models can handle infinite-length sequences without
    performance degradation.
    
    Cache structure:
        [initial tokens (0:start_size)] + [recent tokens (seq_len-recent_size:seq_len)]
    
    Args:
        past_key_values: KV cache, shape (batch, heads, seq_len, head_dim)
        start_size: Number of initial tokens to keep as attention sinks (default: 4)
                   The paper shows 4 initial tokens suffice for most models.
        recent_size: Number of recent tokens to keep in sliding window (default: 508)
                    Effective context = start_size + recent_size
        skip_layers: Layer indices to skip compression (default: empty)
        **kwargs: Additional arguments (ignored, for interface compatibility)
    
    Returns:
        Compressed KV cache with at most (start_size + recent_size) tokens
    
    Note:
        - Total cache size = start_size + recent_size
        - For models with RoPE, position IDs should be adjusted after compression
          to maintain relative positions. This function handles the KV cache only;
          position adjustments should be handled separately if needed.
        - Default values (4 + 508 = 512) match typical cache sizes in the paper.
    
    Example:
        >>> # Keep 4 initial tokens + 508 recent = 512 total
        >>> compressed = streaming_llm_compress(
        ...     past_key_values,
        ...     start_size=4,
        ...     recent_size=508
        ... )
        
        >>> # Larger context window
        >>> compressed = streaming_llm_compress(
        ...     past_key_values,
        ...     start_size=4,
        ...     recent_size=2044  # 4 + 2044 = 2048 total
        ... )
    
    Reference:
        Xiao et al., "Efficient Streaming Language Models with Attention Sinks"
        ICLR 2024
    """
    # Convert to list format if needed
    past_key_values = list(normalize_kv_cache(past_key_values))
    
    if not past_key_values:
        return past_key_values
    
    cache_size = start_size + recent_size
    
    for layer_idx, (keys, values) in enumerate(past_key_values):
        seq_len = keys.size(2)
        
        # No compression needed if within cache size
        if seq_len <= cache_size:
            continue
        
        # Skip specified layers
        if layer_idx in skip_layers:
            continue
        
        # StreamingLLM: keep initial tokens (attention sinks) + recent tokens
        # Initial tokens: [0, start_size)
        # Recent tokens: [seq_len - recent_size, seq_len)
        
        initial_keys = keys[:, :, :start_size, :]
        initial_values = values[:, :, :start_size, :]
        
        recent_keys = keys[:, :, -recent_size:, :]
        recent_values = values[:, :, -recent_size:, :]
        
        # Concatenate: initial + recent
        compressed_keys = torch.cat([initial_keys, recent_keys], dim=2)
        compressed_values = torch.cat([initial_values, recent_values], dim=2)
        
        past_key_values[layer_idx] = (compressed_keys, compressed_values)
    
    return past_key_values


def evict_for_space(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache],
    num_coming: int,
    start_size: int = 4,
    recent_size: int = 508,
    skip_layers: List[int] = [],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Evict tokens to make space for incoming tokens.
    
    This is useful during generation when you know how many new tokens
    will be added and want to proactively make space.
    
    Args:
        past_key_values: KV cache
        num_coming: Number of incoming tokens
        start_size: Number of initial tokens to keep
        recent_size: Number of recent tokens to keep
        skip_layers: Layer indices to skip compression
    
    Returns:
        Compressed KV cache with space for incoming tokens
    """
    past_key_values = list(normalize_kv_cache(past_key_values))
    
    if not past_key_values:
        return past_key_values
    
    cache_size = start_size + recent_size
    
    for layer_idx, (keys, values) in enumerate(past_key_values):
        seq_len = keys.size(2)
        
        # Check if eviction is needed
        if seq_len + num_coming <= cache_size:
            continue
        
        if layer_idx in skip_layers:
            continue
        
        # Evict to make space: keep start_size + (recent_size - num_coming)
        effective_recent = recent_size - num_coming
        if effective_recent <= 0:
            effective_recent = recent_size
        
        initial_keys = keys[:, :, :start_size, :]
        initial_values = values[:, :, :start_size, :]
        
        recent_keys = keys[:, :, -effective_recent:, :]
        recent_values = values[:, :, -effective_recent:, :]
        
        compressed_keys = torch.cat([initial_keys, recent_keys], dim=2)
        compressed_values = torch.cat([initial_values, recent_values], dim=2)
        
        past_key_values[layer_idx] = (compressed_keys, compressed_values)
    
    return past_key_values


__all__ = ['streaming_llm_compress', 'evict_for_space']

