"""
Adaptive L2: Dynamic Sequence-Length Aware KV Cache Compression

This module implements adaptive compression that adjusts its behavior based
on the current sequence length. Short sequences are not compressed, medium
sequences get gentle compression, and long sequences get aggressive compression.

This approach provides a smooth transition rather than sudden cache eviction,
which can help maintain quality while still achieving compression benefits
for long sequences.
"""

from typing import List, Tuple, Union
import torch
from transformers import DynamicCache

from ..utils import normalize_kv_cache


def adaptive_l2_compress(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache],
    target_size: int = 512,
    soft_limit: int = 256,
    hard_limit: int = 1024,
    keep_ratio_min: float = 0.3,
    keep_ratio_max: float = 0.9,
    skip_layers: List[int] = [],
    **kwargs
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Adaptive L2 compression: dynamic compression based on sequence length.
    
    Compression behavior:
    - seq_len <= soft_limit: No compression (keep all tokens)
    - soft_limit < seq_len <= hard_limit: Gradual compression
      keep_ratio decreases linearly from keep_ratio_max to keep_ratio_min
    - seq_len > hard_limit: Compress to target_size with L2-based selection
    
    Args:
        past_key_values: KV cache, shape (batch, heads, seq_len, head_dim)
        target_size: Target cache size for long sequences (default: 512)
        soft_limit: Below this length, no compression (default: 256)
        hard_limit: Above this length, compress to target_size (default: 1024)
        keep_ratio_min: Minimum keep ratio at hard_limit (default: 0.3)
        keep_ratio_max: Maximum keep ratio at soft_limit (default: 0.9)
        skip_layers: Layer indices to skip compression (default: empty)
        **kwargs: Additional arguments (ignored, for interface compatibility)
    
    Returns:
        Compressed KV cache with adaptive size based on input length
    
    Example:
        >>> # Adaptive compression: no change for short, gradual for medium, aggressive for long
        >>> compressed = adaptive_l2_compress(
        ...     past_key_values,
        ...     target_size=512,
        ...     soft_limit=256,
        ...     hard_limit=1024
        ... )
    """
    # Convert to list format if needed
    past_key_values = list(normalize_kv_cache(past_key_values))
    
    if not past_key_values:
        return past_key_values
    
    for layer_idx, (keys, values) in enumerate(past_key_values):
        seq_len = keys.size(2)
        
        # Skip specified layers
        if layer_idx in skip_layers:
            continue
        
        # Determine compression strategy based on sequence length
        if seq_len <= soft_limit:
            # No compression needed for short sequences
            continue
        
        batch_size, num_heads, _, head_dim = keys.shape
        
        if seq_len > hard_limit:
            # Hard limit exceeded: compress to target_size
            # Use StreamingLLM-style approach with L2 selection
            
            if seq_len <= target_size:
                continue
            
            # Keep some attention sinks + L2-selected middle + recent
            start_size = 4
            recent_size = target_size // 2
            middle_to_keep = target_size - start_size - recent_size
            
            if middle_to_keep <= 0:
                # Just keep recent tokens
                compressed_keys = keys[:, :, -target_size:, :]
                compressed_values = values[:, :, -target_size:, :]
                past_key_values[layer_idx] = (compressed_keys, compressed_values)
                continue
            
            # Get regions
            middle_start = start_size
            middle_end = seq_len - recent_size
            
            if middle_end <= middle_start:
                sink_keys = keys[:, :, :start_size, :]
                sink_values = values[:, :, :start_size, :]
                recent_keys = keys[:, :, -(target_size - start_size):, :]
                recent_values = values[:, :, -(target_size - start_size):, :]
                
                compressed_keys = torch.cat([sink_keys, recent_keys], dim=2)
                compressed_values = torch.cat([sink_values, recent_values], dim=2)
                past_key_values[layer_idx] = (compressed_keys, compressed_values)
                continue
            
            # Get sinks
            sink_keys = keys[:, :, :start_size, :]
            sink_values = values[:, :, :start_size, :]
            
            # Select from middle by L2 norm
            middle_keys = keys[:, :, middle_start:middle_end, :]
            middle_values = values[:, :, middle_start:middle_end, :]
            middle_len = middle_keys.size(2)
            
            num_to_keep = min(middle_to_keep, middle_len)
            
            token_norms = torch.norm(middle_keys, p=2, dim=-1)
            sorted_indices = token_norms.argsort(dim=-1)
            indices_to_keep = sorted_indices[:, :, :num_to_keep]
            indices_to_keep_sorted, _ = torch.sort(indices_to_keep, dim=-1)
            
            indices_expanded = indices_to_keep_sorted.unsqueeze(-1).expand(
                batch_size, num_heads, num_to_keep, head_dim
            )
            
            selected_middle_keys = torch.gather(middle_keys, dim=2, index=indices_expanded)
            selected_middle_values = torch.gather(middle_values, dim=2, index=indices_expanded)
            
            # Get recent
            recent_keys = keys[:, :, -recent_size:, :]
            recent_values = values[:, :, -recent_size:, :]
            
            compressed_keys = torch.cat([sink_keys, selected_middle_keys, recent_keys], dim=2)
            compressed_values = torch.cat([sink_values, selected_middle_values, recent_values], dim=2)
            
            past_key_values[layer_idx] = (compressed_keys, compressed_values)
        
        else:
            # Soft limit < seq_len <= hard_limit: gradual compression
            # Calculate adaptive keep_ratio based on position in range
            progress = (seq_len - soft_limit) / (hard_limit - soft_limit)
            keep_ratio = keep_ratio_max - progress * (keep_ratio_max - keep_ratio_min)
            
            tokens_to_keep = int(seq_len * keep_ratio)
            tokens_to_keep = max(tokens_to_keep, soft_limit)  # Don't go below soft_limit
            
            if tokens_to_keep >= seq_len:
                continue
            
            # Protect recent tokens (last 20%)
            protected_recent = int(tokens_to_keep * 0.2)
            tokens_from_history = tokens_to_keep - protected_recent
            
            if tokens_from_history <= 0:
                # Just keep recent
                compressed_keys = keys[:, :, -tokens_to_keep:, :]
                compressed_values = values[:, :, -tokens_to_keep:, :]
                past_key_values[layer_idx] = (compressed_keys, compressed_values)
                continue
            
            # Selection zone: everything except protected recent
            selection_end = seq_len - protected_recent
            
            if selection_end <= tokens_from_history:
                continue
            
            selection_keys = keys[:, :, :selection_end, :]
            selection_values = values[:, :, :selection_end, :]
            
            # Select by L2 norm
            token_norms = torch.norm(selection_keys, p=2, dim=-1)
            sorted_indices = token_norms.argsort(dim=-1)
            indices_to_keep = sorted_indices[:, :, :tokens_from_history]
            indices_to_keep_sorted, _ = torch.sort(indices_to_keep, dim=-1)
            
            indices_expanded = indices_to_keep_sorted.unsqueeze(-1).expand(
                batch_size, num_heads, tokens_from_history, head_dim
            )
            
            selected_keys = torch.gather(selection_keys, dim=2, index=indices_expanded)
            selected_values = torch.gather(selection_values, dim=2, index=indices_expanded)
            
            # Get protected recent
            recent_keys = keys[:, :, -protected_recent:, :]
            recent_values = values[:, :, -protected_recent:, :]
            
            compressed_keys = torch.cat([selected_keys, recent_keys], dim=2)
            compressed_values = torch.cat([selected_values, recent_values], dim=2)
            
            past_key_values[layer_idx] = (compressed_keys, compressed_values)
    
    return past_key_values


__all__ = ['adaptive_l2_compress']

