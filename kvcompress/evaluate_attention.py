"""
Evaluation Module with Attention Score Support

This module provides evaluation functions that can access attention scores
for advanced compression methods like H2O.

The key difference from standard evaluation is:
- Uses output_attentions=True in model forward pass
- Passes attention scores to compression function
- Supports H2OAttentionManager for accumulated attention tracking
"""

import time
from typing import Callable, Dict, List, Optional, Union
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import DynamicCache

from .utils import to_dynamic_cache, normalize_kv_cache
from .methods.h2o_attention import H2OAttentionManager, h2o_attention_compress


def evaluate_with_attention_compression(
    model,
    tokenizer,
    text: str,
    h2o_manager: Optional[H2OAttentionManager] = None,
    start_size: int = 4,
    heavy_hitter_size: int = 64,
    recent_size: int = 444,
    max_tokens: int = 3000,
    skip_layers: List[int] = [0, 1],
    device: Optional[torch.device] = None,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Evaluate PPL and Accuracy with H2O attention-based compression.
    
    This function uses real attention scores from the model to identify
    heavy hitters, implementing the true H2O algorithm.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text to evaluate
        h2o_manager: H2OAttentionManager for tracking accumulated attention
                    If None, one will be created automatically
        start_size: Number of initial tokens (attention sinks)
        heavy_hitter_size: Number of heavy hitters to keep
        recent_size: Number of recent tokens to keep
        max_tokens: Maximum number of tokens to evaluate
        skip_layers: Layer indices to skip compression
        device: Device to use
        show_progress: Whether to show progress bar
    
    Returns:
        Dict containing perplexity, accuracy, timing metrics, etc.
    
    Example:
        >>> from kvcompress.methods.h2o_attention import create_h2o_manager_from_model
        >>> manager = create_h2o_manager_from_model(model, heavy_hitter_size=64)
        >>> results = evaluate_with_attention_compression(
        ...     model, tokenizer, text,
        ...     h2o_manager=manager,
        ...     heavy_hitter_size=64
        ... )
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Create manager if not provided
    if h2o_manager is None:
        num_layers = getattr(model.config, 'num_hidden_layers', 32)
        num_heads = getattr(model.config, 'num_attention_heads', 32)
        h2o_manager = H2OAttentionManager(
            start_size=start_size,
            heavy_hitter_size=heavy_hitter_size,
            recent_size=recent_size,
            num_layers=num_layers,
            num_heads=num_heads,
            device=device,
        )
    else:
        # Reset manager for new sequence
        h2o_manager.reset()
    
    # Tokenize input
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids[:, :max_tokens].to(device)
    seq_len = input_ids.shape[1]
    
    if seq_len < 2:
        return {
            "perplexity": float('inf'),
            "accuracy": 0.0,
            "num_tokens": 0,
            "final_cache_size": 0,
            "ttft": 0.0,
            "tpot": 0.0,
            "throughput": 0.0,
            "total_time": 0.0,
        }
    
    # Loss function
    loss_fn = CrossEntropyLoss(reduction="none")
    
    # Initialize
    past_key_values = None
    nlls = []
    num_correct = []
    
    # Timing
    ttft = None
    token_times = []
    
    model.eval()
    
    # Progress bar
    token_range = range(seq_len - 1)
    if show_progress:
        token_range = tqdm(token_range, desc="H2O-Attention Eval")
    
    total_start = time.perf_counter()
    
    total_cache_size = start_size + heavy_hitter_size + recent_size
    
    with torch.inference_mode():
        for idx in token_range:
            token_start = time.perf_counter()
            
            current_token = input_ids[:, idx:idx+1]
            target = input_ids[:, idx+1:idx+2].view(-1)
            
            # Forward pass WITH attention outputs
            outputs = model(
                current_token,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,  # Key difference!
            )
            
            # Get logits
            logits = outputs.logits[:, -1, :].view(-1, model.config.vocab_size)
            
            # Calculate NLL and accuracy
            nll = loss_fn(logits, target)
            nlls.append(nll.item())
            
            predicted = torch.argmax(logits, dim=-1)
            num_correct.append((predicted == target).int().item())
            
            # Get KV cache and attention scores
            past_key_values = outputs.past_key_values
            attention_scores = outputs.attentions  # Tuple of (batch, heads, q_len, k_len)
            
            # Update H2O manager with new attention scores
            h2o_manager.update_attention_scores(attention_scores, skip_layers)
            
            # Apply H2O compression with real attention scores
            if past_key_values is not None:
                kv_list = list(normalize_kv_cache(past_key_values))
                current_cache_len = kv_list[0][0].size(2) if kv_list else 0
                
                # Only compress if cache exceeds limit
                if current_cache_len > total_cache_size:
                    compressed_kv = h2o_attention_compress(
                        kv_list,
                        attention_scores=attention_scores,
                        h2o_manager=h2o_manager,
                        start_size=start_size,
                        heavy_hitter_size=heavy_hitter_size,
                        recent_size=recent_size,
                        skip_layers=skip_layers,
                    )
                    past_key_values = to_dynamic_cache(compressed_kv)
                    
                    # Reset accumulated attention after compression
                    # (positions have changed)
                    h2o_manager.reset()
            
            # Record timing
            token_time = time.perf_counter() - token_start
            token_times.append(token_time)
            
            if ttft is None:
                ttft = token_time
            
            # Update progress
            if show_progress:
                current_ppl = torch.exp(torch.tensor(nlls).mean()).item()
                current_acc = sum(num_correct) / len(num_correct)
                token_range.set_description(
                    f"H2O-Attn | PPL: {current_ppl:.2f}, Acc: {current_acc:.2%}"
                )
    
    total_time = time.perf_counter() - total_start
    
    # Calculate final metrics
    perplexity = torch.exp(torch.tensor(nlls).mean()).item()
    accuracy = sum(num_correct) / len(num_correct)
    
    num_tokens = len(nlls)
    tpot = sum(token_times[1:]) / (num_tokens - 1) if num_tokens > 1 else ttft
    throughput = num_tokens / total_time if total_time > 0 else 0.0
    
    # Get final cache size
    final_cache_size = 0
    if past_key_values is not None:
        kv_list = list(normalize_kv_cache(past_key_values))
        if kv_list:
            for layer_idx, (k, v) in enumerate(kv_list):
                if layer_idx not in skip_layers:
                    final_cache_size = k.size(2)
                    break
            if final_cache_size == 0:
                final_cache_size = kv_list[0][0].size(2)
    
    return {
        "perplexity": perplexity,
        "accuracy": accuracy,
        "num_tokens": num_tokens,
        "final_cache_size": final_cache_size,
        "ttft": ttft if ttft else 0.0,
        "tpot": tpot,
        "throughput": throughput,
        "total_time": total_time,
    }


def compare_h2o_methods(
    model,
    tokenizer,
    text: str,
    max_tokens: int = 2000,
    heavy_hitter_sizes: List[int] = [32, 64, 128],
    skip_layers: List[int] = [0, 1],
    device: Optional[torch.device] = None,
) -> List[Dict]:
    """
    Compare H2O-L2 (approximation) vs H2O-Attention (real attention).
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text
        max_tokens: Maximum tokens to evaluate
        heavy_hitter_sizes: List of heavy hitter sizes to test
        skip_layers: Layers to skip
        device: Device to use
    
    Returns:
        List of result dictionaries
    """
    from .methods import h2o_l2_compress
    from .evaluate import evaluate_with_compression
    
    if device is None:
        device = next(model.parameters()).device
    
    results = []
    
    # Baseline (no compression)
    print("\n" + "="*60)
    print("Testing: Baseline (no compression)")
    print("="*60)
    
    baseline = evaluate_with_compression(
        model, tokenizer, text,
        compress_fn=None,
        max_tokens=max_tokens,
        device=device,
    )
    baseline['method'] = 'baseline'
    results.append(baseline)
    print(f"  PPL: {baseline['perplexity']:.2f}, Acc: {baseline['accuracy']:.2%}")
    
    for hh_size in heavy_hitter_sizes:
        total_size = 4 + hh_size + (512 - 4 - hh_size)  # Total = 512
        recent = 512 - 4 - hh_size
        
        # H2O-L2 (approximation)
        print("\n" + "="*60)
        print(f"Testing: H2O-L2 (hh={hh_size}, total=512)")
        print("="*60)
        
        h2o_l2_result = evaluate_with_compression(
            model, tokenizer, text,
            compress_fn=h2o_l2_compress,
            compress_kwargs={
                "start_size": 4,
                "heavy_hitter_size": hh_size,
                "recent_size": recent,
            },
            max_tokens=max_tokens,
            skip_layers=skip_layers,
            device=device,
        )
        h2o_l2_result['method'] = f'h2o_l2_hh{hh_size}'
        results.append(h2o_l2_result)
        print(f"  PPL: {h2o_l2_result['perplexity']:.2f}, Acc: {h2o_l2_result['accuracy']:.2%}")
        
        # H2O-Attention (real attention)
        print("\n" + "="*60)
        print(f"Testing: H2O-Attention (hh={hh_size}, total=512)")
        print("="*60)
        
        h2o_attn_result = evaluate_with_attention_compression(
            model, tokenizer, text,
            start_size=4,
            heavy_hitter_size=hh_size,
            recent_size=recent,
            max_tokens=max_tokens,
            skip_layers=skip_layers,
            device=device,
        )
        h2o_attn_result['method'] = f'h2o_attention_hh{hh_size}'
        results.append(h2o_attn_result)
        print(f"  PPL: {h2o_attn_result['perplexity']:.2f}, Acc: {h2o_attn_result['accuracy']:.2%}")
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Method':<25} {'PPL':>10} {'Acc':>10} {'Throughput':>12} {'Cache':>8}")
    print("-"*80)
    
    for r in results:
        print(f"{r['method']:<25} {r['perplexity']:>10.2f} {r['accuracy']:>10.2%} "
              f"{r['throughput']:>12.2f} {r['final_cache_size']:>8}")
    
    return results


__all__ = [
    'evaluate_with_attention_compression',
    'compare_h2o_methods',
]


