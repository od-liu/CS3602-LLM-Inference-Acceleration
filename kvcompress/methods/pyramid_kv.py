"""
Pyramid KV: Layer-wise Adaptive KV Cache Compression

This module implements layer-wise adaptive compression where different layers
get different compression ratios. The key insight is that lower layers capture
local patterns (need more tokens) while higher layers are more redundant.

Pyramid profile:
- Lower layers: Keep more tokens (higher cache size)
- Higher layers: Keep fewer tokens (aggressive compression)

This follows the observation that attention patterns in lower layers tend to
be more local and require more context, while higher layers are more abstract.

Reference: PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling
           https://arxiv.org/abs/2406.02069
"""

from typing import List, Tuple, Union, Literal
import torch
from transformers import DynamicCache

from ..utils import normalize_kv_cache


def pyramid_kv_compress(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache],
    base_size: int = 512,
    layer_decay: float = 0.9,
    min_size: int = 64,
    profile: Literal["linear", "exponential", "constant"] = "exponential",
    skip_layers: List[int] = [],
    **kwargs
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Pyramid KV compression: layer-wise adaptive cache size with L2 selection.
    
    This method applies different compression ratios to different layers:
    - Layer 0: base_size tokens
    - Layer i: base_size * (layer_decay ^ i) tokens
    
    Tokens are selected based on L2 norm (lower = more important).
    
    Args:
        past_key_values: KV cache, shape (batch, heads, seq_len, head_dim)
        base_size: Cache size for the first layer (default: 512)
        layer_decay: Decay factor per layer (default: 0.9)
                    Layer i gets base_size * (layer_decay ^ i) tokens
        min_size: Minimum cache size for any layer (default: 64)
        profile: How to compute layer sizes
                - "exponential": size = base_size * (decay ^ layer_idx)
                - "linear": size = base_size - layer_idx * (base_size - min_size) / num_layers
                - "constant": all layers get base_size
        skip_layers: Layer indices to skip compression (default: empty)
        **kwargs: Additional arguments (ignored, for interface compatibility)
    
    Returns:
        Compressed KV cache with layer-specific sizes
    
    Example:
        >>> # Exponential decay: layer 0=512, layer 1=460, layer 2=414, ...
        >>> compressed = pyramid_kv_compress(
        ...     past_key_values,
        ...     base_size=512,
        ...     layer_decay=0.9
        ... )
        
        >>> # Linear decay from 512 to 128 across layers
        >>> compressed = pyramid_kv_compress(
        ...     past_key_values,
        ...     base_size=512,
        ...     min_size=128,
        ...     profile="linear"
        ... )
    """
    # Convert to list format if needed
    past_key_values = list(normalize_kv_cache(past_key_values))
    
    if not past_key_values:
        return past_key_values
    
    num_layers = len(past_key_values)
    
    # Pre-compute cache sizes for each layer
    layer_sizes = []
    for layer_idx in range(num_layers):
        if profile == "exponential":
            size = int(base_size * (layer_decay ** layer_idx))
        elif profile == "linear":
            decay_per_layer = (base_size - min_size) / max(num_layers - 1, 1)
            size = int(base_size - layer_idx * decay_per_layer)
        else:  # constant
            size = base_size
        
        # Enforce minimum size
        size = max(size, min_size)
        layer_sizes.append(size)
    
    for layer_idx, (keys, values) in enumerate(past_key_values):
        seq_len = keys.size(2)
        target_size = layer_sizes[layer_idx]
        
        # No compression needed if within limit
        if seq_len <= target_size:
            continue
        
        # Skip specified layers
        if layer_idx in skip_layers:
            continue
        
        batch_size, num_heads, _, head_dim = keys.shape
        
        # Use StreamingLLM-style approach with L2 selection for middle tokens
        # Keep some initial tokens (attention sinks) + selected middle + recent
        start_size = min(4, target_size // 8)  # Small attention sink
        recent_size = target_size // 2  # Half for recent tokens
        middle_to_keep = target_size - start_size - recent_size
        
        if middle_to_keep <= 0:
            # Fall back to simple recent-only for very small target sizes
            compressed_keys = keys[:, :, -target_size:, :]
            compressed_values = values[:, :, -target_size:, :]
            past_key_values[layer_idx] = (compressed_keys, compressed_values)
            continue
        
        # Get regions
        middle_start = start_size
        middle_end = seq_len - recent_size
        
        if middle_end <= middle_start:
            # No room for middle selection
            sink_keys = keys[:, :, :start_size, :]
            sink_values = values[:, :, :start_size, :]
            recent_keys = keys[:, :, -(target_size - start_size):, :]
            recent_values = values[:, :, -(target_size - start_size):, :]
            
            compressed_keys = torch.cat([sink_keys, recent_keys], dim=2)
            compressed_values = torch.cat([sink_values, recent_values], dim=2)
            past_key_values[layer_idx] = (compressed_keys, compressed_values)
            continue
        
        # Get attention sinks
        sink_keys = keys[:, :, :start_size, :]
        sink_values = values[:, :, :start_size, :]
        
        # Get middle tokens and select by L2 norm
        middle_keys = keys[:, :, middle_start:middle_end, :]
        middle_values = values[:, :, middle_start:middle_end, :]
        middle_len = middle_keys.size(2)
        
        num_to_keep = min(middle_to_keep, middle_len)
        
        if num_to_keep > 0 and middle_len > 0:
            # Calculate L2 norms
            token_norms = torch.norm(middle_keys, p=2, dim=-1)
            
            # Select lowest norm tokens (most important)
            sorted_indices = token_norms.argsort(dim=-1)
            indices_to_keep = sorted_indices[:, :, :num_to_keep]
            
            # Maintain temporal order
            indices_to_keep_sorted, _ = torch.sort(indices_to_keep, dim=-1)
            
            # Expand and gather
            indices_expanded = indices_to_keep_sorted.unsqueeze(-1).expand(
                batch_size, num_heads, num_to_keep, head_dim
            )
            
            selected_middle_keys = torch.gather(middle_keys, dim=2, index=indices_expanded)
            selected_middle_values = torch.gather(middle_values, dim=2, index=indices_expanded)
        else:
            selected_middle_keys = middle_keys[:, :, :0, :]  # Empty tensor
            selected_middle_values = middle_values[:, :, :0, :]
        
        # Get recent tokens
        recent_keys = keys[:, :, -recent_size:, :]
        recent_values = values[:, :, -recent_size:, :]
        
        # Concatenate all parts
        compressed_keys = torch.cat([sink_keys, selected_middle_keys, recent_keys], dim=2)
        compressed_values = torch.cat([sink_values, selected_middle_values, recent_values], dim=2)
        
        past_key_values[layer_idx] = (compressed_keys, compressed_values)
    
    return past_key_values


__all__ = ['pyramid_kv_compress']

