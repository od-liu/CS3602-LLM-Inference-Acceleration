"""
H2O-Attention: True Heavy-Hitter Oracle with Real Attention Scores

This module implements the original H2O algorithm using actual attention scores
from the model, rather than L2 norm approximation.

The H2O algorithm identifies "heavy hitter" tokens based on accumulated attention
scores - tokens that consistently receive high attention across multiple queries
are considered important and retained in the cache.

Key Features:
1. Uses real attention scores from model forward pass
2. Accumulates attention importance across tokens (heavy hitter detection)
3. Combines attention sinks + heavy hitters + recent window
4. Per-head heavy hitter selection for better quality

Reference: H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs
           https://arxiv.org/abs/2306.14048
"""

from typing import List, Tuple, Union, Optional, Dict
import torch
from transformers import DynamicCache

from ..utils import normalize_kv_cache


class H2OAttentionManager:
    """
    Manager class for H2O algorithm with accumulated attention scores.
    
    This class maintains the accumulated attention scores across tokens
    and provides methods to identify heavy hitters based on actual attention.
    """
    
    def __init__(
        self,
        start_size: int = 4,
        heavy_hitter_size: int = 64,
        recent_size: int = 444,
        num_layers: int = 32,
        num_heads: int = 32,
        decay_factor: float = 0.9,
        device: torch.device = None,
    ):
        """
        Initialize H2O attention manager.
        
        Args:
            start_size: Number of initial tokens (attention sinks) to always keep
            heavy_hitter_size: Number of heavy hitters to keep from middle region
            recent_size: Number of recent tokens to always keep
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
            decay_factor: Exponential decay for accumulated attention scores (0-1)
                         Higher values = more weight on historical importance
            device: Device for tensors
        """
        self.start_size = start_size
        self.heavy_hitter_size = heavy_hitter_size
        self.recent_size = recent_size
        self.total_cache_size = start_size + heavy_hitter_size + recent_size
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.decay_factor = decay_factor
        self.device = device
        
        # Accumulated attention scores per layer and head
        # Shape: {layer_idx: Tensor of shape (batch, heads, max_seq_len)}
        self.accumulated_attention: Dict[int, torch.Tensor] = {}
        
        # Track which token positions are currently in the cache
        self.token_positions: Dict[int, torch.Tensor] = {}
        
        self.current_seq_len = 0
    
    def reset(self):
        """Reset accumulated attention scores for new sequence."""
        self.accumulated_attention = {}
        self.token_positions = {}
        self.current_seq_len = 0
    
    def update_attention_scores(
        self,
        attentions: Tuple[torch.Tensor, ...],
        skip_layers: List[int] = [],
    ):
        """
        Update accumulated attention scores with new attention weights.
        
        Args:
            attentions: Tuple of attention tensors from model output
                       Each tensor has shape (batch, heads, query_len, key_len)
            skip_layers: Layers to skip updating
        """
        if attentions is None:
            return
        
        for layer_idx, attn in enumerate(attentions):
            if layer_idx in skip_layers:
                continue
            
            # Skip if attention is None (some models don't return all layers)
            if attn is None:
                continue
            
            # attn shape: (batch, heads, query_len, key_len)
            # For autoregressive generation, query_len is typically 1
            # We sum attention over query positions to get importance per key
            
            batch_size, num_heads, query_len, key_len = attn.shape
            
            # Sum attention scores across query positions
            # This gives importance of each key position
            attn_importance = attn.sum(dim=2)  # (batch, heads, key_len)
            
            if layer_idx not in self.accumulated_attention:
                # Initialize with zeros
                self.accumulated_attention[layer_idx] = torch.zeros(
                    batch_size, num_heads, key_len,
                    device=attn.device, dtype=attn.dtype
                )
            else:
                # Decay existing scores and handle size changes
                existing = self.accumulated_attention[layer_idx]
                existing_len = existing.size(-1)
                
                if existing_len < key_len:
                    # Extend with zeros for new positions
                    padding = torch.zeros(
                        batch_size, num_heads, key_len - existing_len,
                        device=attn.device, dtype=attn.dtype
                    )
                    self.accumulated_attention[layer_idx] = torch.cat(
                        [existing * self.decay_factor, padding], dim=-1
                    )
                elif existing_len > key_len:
                    # This can happen after compression - need to remap
                    # For simplicity, we reset in this case
                    self.accumulated_attention[layer_idx] = torch.zeros(
                        batch_size, num_heads, key_len,
                        device=attn.device, dtype=attn.dtype
                    )
                else:
                    self.accumulated_attention[layer_idx] = existing * self.decay_factor
            
            # Add new attention importance
            self.accumulated_attention[layer_idx] = (
                self.accumulated_attention[layer_idx] + attn_importance
            )
            
            # Update current sequence length
            self.current_seq_len = key_len
    
    def get_heavy_hitter_indices(
        self,
        layer_idx: int,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Get indices of heavy hitter tokens for a specific layer.
        
        Args:
            layer_idx: Layer index
            seq_len: Current sequence length
        
        Returns:
            Tensor of indices to keep (sorted in temporal order)
        """
        if layer_idx not in self.accumulated_attention:
            # No attention data, return evenly spaced indices
            middle_start = self.start_size
            middle_end = seq_len - self.recent_size
            if middle_end <= middle_start:
                return torch.tensor([], dtype=torch.long)
            
            middle_len = middle_end - middle_start
            step = max(1, middle_len // self.heavy_hitter_size)
            indices = torch.arange(0, middle_len, step)[:self.heavy_hitter_size]
            return indices
        
        acc_attn = self.accumulated_attention[layer_idx]
        batch_size, num_heads, attn_len = acc_attn.shape
        
        # Define middle region
        middle_start = self.start_size
        middle_end = min(seq_len, attn_len) - self.recent_size
        
        if middle_end <= middle_start:
            return torch.tensor([], dtype=torch.long, device=acc_attn.device)
        
        # Get attention scores for middle region
        middle_attn = acc_attn[:, :, middle_start:middle_end]  # (batch, heads, middle_len)
        
        # Aggregate across heads (sum or mean)
        # Using sum to identify tokens important to ANY head
        head_aggregated = middle_attn.sum(dim=1)  # (batch, middle_len)
        
        # For batch=1, squeeze
        if batch_size == 1:
            head_aggregated = head_aggregated.squeeze(0)  # (middle_len,)
        
        # Get top-k indices (highest accumulated attention)
        middle_len = middle_end - middle_start
        k = min(self.heavy_hitter_size, middle_len)
        
        _, top_indices = torch.topk(head_aggregated, k, dim=-1)
        
        # Sort to maintain temporal order
        top_indices_sorted, _ = torch.sort(top_indices, dim=-1)
        
        return top_indices_sorted


def h2o_attention_compress(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache],
    attention_scores: Optional[Tuple[torch.Tensor, ...]] = None,
    h2o_manager: Optional[H2OAttentionManager] = None,
    start_size: int = 4,
    heavy_hitter_size: int = 64,
    recent_size: int = 444,
    skip_layers: List[int] = [],
    **kwargs
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    H2O compression using real attention scores.
    
    This function implements the true H2O algorithm by:
    1. Using accumulated attention scores to identify heavy hitters
    2. Keeping attention sinks (initial tokens)
    3. Keeping heavy hitters from middle region
    4. Keeping recent tokens in sliding window
    
    Args:
        past_key_values: KV cache, shape (batch, heads, seq_len, head_dim)
        attention_scores: Tuple of attention tensors from model output
                         Each tensor has shape (batch, heads, query_len, key_len)
                         If None, falls back to L2 norm-based selection
        h2o_manager: H2OAttentionManager instance for tracking accumulated attention
                    If None, a temporary one is created (less effective)
        start_size: Number of initial tokens to keep as attention sinks
        heavy_hitter_size: Number of heavy hitters to keep from middle
        recent_size: Number of recent tokens to keep
        skip_layers: Layer indices to skip compression
        **kwargs: Additional arguments (ignored)
    
    Returns:
        Compressed KV cache
    
    Example:
        >>> # Create manager once per sequence
        >>> manager = H2OAttentionManager(start_size=4, heavy_hitter_size=64, recent_size=444)
        >>> 
        >>> # During generation
        >>> outputs = model(input_ids, output_attentions=True, use_cache=True)
        >>> manager.update_attention_scores(outputs.attentions)
        >>> compressed_kv = h2o_attention_compress(
        ...     outputs.past_key_values,
        ...     attention_scores=outputs.attentions,
        ...     h2o_manager=manager
        ... )
    """
    # Convert to list format
    past_key_values = list(normalize_kv_cache(past_key_values))
    
    if not past_key_values:
        return past_key_values
    
    total_cache_size = start_size + heavy_hitter_size + recent_size
    
    # Update manager with new attention scores if provided
    if h2o_manager is not None and attention_scores is not None:
        h2o_manager.update_attention_scores(attention_scores, skip_layers)
    
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
        middle_start = start_size
        middle_end = seq_len - recent_size
        
        # If middle region is too small, fall back to StreamingLLM behavior
        if middle_end <= middle_start:
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
        
        # Get middle region
        middle_keys = keys[:, :, middle_start:middle_end, :]
        middle_values = values[:, :, middle_start:middle_end, :]
        middle_len = middle_keys.size(2)
        
        # Get heavy hitter indices
        if h2o_manager is not None:
            # Use attention-based selection
            heavy_indices = h2o_manager.get_heavy_hitter_indices(layer_idx, seq_len)
            num_to_keep = min(len(heavy_indices), heavy_hitter_size, middle_len)
            
            if num_to_keep > 0 and len(heavy_indices) > 0:
                heavy_indices = heavy_indices[:num_to_keep]
                
                # Ensure indices are valid and on correct device
                heavy_indices = heavy_indices.clamp(0, middle_len - 1).to(keys.device)
                
                # Expand for gather
                indices_expanded = heavy_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                indices_expanded = indices_expanded.expand(
                    batch_size, num_heads, num_to_keep, head_dim
                )
                
                heavy_keys = torch.gather(middle_keys, dim=2, index=indices_expanded)
                heavy_values = torch.gather(middle_values, dim=2, index=indices_expanded)
            else:
                heavy_keys = middle_keys[:, :, :0, :]
                heavy_values = middle_values[:, :, :0, :]
        else:
            # Fallback: use L2 norm-based selection (like H2O-L2)
            token_norms = torch.norm(middle_keys, p=2, dim=-1)
            num_to_keep = min(heavy_hitter_size, middle_len)
            
            sorted_indices = token_norms.argsort(dim=-1)
            indices_to_keep = sorted_indices[:, :, :num_to_keep]
            indices_to_keep_sorted, _ = torch.sort(indices_to_keep, dim=-1)
            
            indices_expanded = indices_to_keep_sorted.unsqueeze(-1).expand(
                batch_size, num_heads, num_to_keep, head_dim
            )
            
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


def create_h2o_manager_from_model(model, **kwargs) -> H2OAttentionManager:
    """
    Create H2OAttentionManager with parameters from model config.
    
    Args:
        model: The transformer model
        **kwargs: Override parameters (start_size, heavy_hitter_size, recent_size, etc.)
    
    Returns:
        Configured H2OAttentionManager
    """
    config = model.config
    
    num_layers = getattr(config, 'num_hidden_layers', 32)
    num_heads = getattr(config, 'num_attention_heads', 32)
    device = next(model.parameters()).device
    
    return H2OAttentionManager(
        start_size=kwargs.get('start_size', 4),
        heavy_hitter_size=kwargs.get('heavy_hitter_size', 64),
        recent_size=kwargs.get('recent_size', 444),
        num_layers=num_layers,
        num_heads=num_heads,
        decay_factor=kwargs.get('decay_factor', 0.9),
        device=device,
    )


__all__ = [
    'H2OAttentionManager',
    'h2o_attention_compress',
    'create_h2o_manager_from_model',
]

