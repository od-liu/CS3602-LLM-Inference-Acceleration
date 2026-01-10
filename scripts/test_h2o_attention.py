#!/usr/bin/env python3
"""
Test script to compare H2O-L2 (approximation) vs H2O-Attention (real attention).

This script demonstrates the difference between:
1. H2O-L2: Uses L2 norm as a proxy for attention importance
2. H2O-Attention: Uses actual attention scores from the model

Usage:
    python scripts/test_h2o_attention.py --model_id /path/to/model
    python scripts/test_h2o_attention.py --compare
"""

import os
import sys
import argparse
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from kvcompress.evaluate_attention import (
    evaluate_with_attention_compression,
    compare_h2o_methods,
)
from kvcompress.methods.h2o_attention import (
    H2OAttentionManager,
    create_h2o_manager_from_model,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_id: str):
    """Load model and tokenizer."""
    print(f"Loading model: {model_id}")
    device = get_device()
    print(f"Using device: {device}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    except (OSError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
        )
    
    model.to(device)
    model.eval()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device


def load_test_text(max_chars: int = 50000):
    """Load a long text sample for testing."""
    local_path = os.path.join(project_root, "data", "pg19.parquet")
    
    if os.path.exists(local_path):
        dataset = load_dataset("parquet", data_files={'test': local_path}, split="test")
    else:
        dataset = load_dataset("pg19", split="test")
    
    # Get first sample with enough text
    for sample in dataset:
        text = sample.get("text", "")
        if len(text) > 10000:
            return text[:max_chars]
    
    return "The quick brown fox jumps over the lazy dog. " * 1000


def main():
    parser = argparse.ArgumentParser(description="Test H2O with real attention scores")
    parser.add_argument("--model_id", type=str, default="EleutherAI/pythia-2.8b",
                       help="Model ID or path")
    parser.add_argument("--max_tokens", type=int, default=1500,
                       help="Maximum tokens to evaluate")
    parser.add_argument("--heavy_hitter_sizes", type=str, default="32,64,128",
                       help="Comma-separated heavy hitter sizes to test")
    parser.add_argument("--compare", action="store_true",
                       help="Run full comparison between H2O-L2 and H2O-Attention")
    parser.add_argument("--skip_layers", type=str, default="0,1",
                       help="Layers to skip compression")
    
    args = parser.parse_args()
    
    # Parse arguments
    hh_sizes = [int(x) for x in args.heavy_hitter_sizes.split(",")]
    skip_layers = [int(x) for x in args.skip_layers.split(",")]
    
    print("="*70)
    print("H2O-Attention Test: Real Attention Scores vs L2 Approximation")
    print("="*70)
    
    # Load model
    model, tokenizer, device = load_model(args.model_id)
    
    # Load test text
    text = load_test_text()
    print(f"\nLoaded text: {len(text)} characters")
    
    if args.compare:
        # Full comparison
        results = compare_h2o_methods(
            model, tokenizer, text,
            max_tokens=args.max_tokens,
            heavy_hitter_sizes=hh_sizes,
            skip_layers=skip_layers,
            device=device,
        )
        
        # Calculate improvements
        baseline = next(r for r in results if r['method'] == 'baseline')
        print("\n" + "="*80)
        print("ANALYSIS: H2O-Attention vs H2O-L2")
        print("="*80)
        
        for hh in hh_sizes:
            l2_result = next(r for r in results if r['method'] == f'h2o_l2_hh{hh}')
            attn_result = next(r for r in results if r['method'] == f'h2o_attention_hh{hh}')
            
            l2_ppl_change = (l2_result['perplexity'] / baseline['perplexity'] - 1) * 100
            attn_ppl_change = (attn_result['perplexity'] / baseline['perplexity'] - 1) * 100
            
            l2_acc_change = (l2_result['accuracy'] / baseline['accuracy'] - 1) * 100
            attn_acc_change = (attn_result['accuracy'] / baseline['accuracy'] - 1) * 100
            
            print(f"\nHeavy Hitter Size = {hh}:")
            print(f"  H2O-L2:        PPL {l2_ppl_change:+.1f}%, Acc {l2_acc_change:+.1f}%")
            print(f"  H2O-Attention: PPL {attn_ppl_change:+.1f}%, Acc {attn_acc_change:+.1f}%")
            
            improvement = l2_ppl_change - attn_ppl_change
            if improvement > 0:
                print(f"  → H2O-Attention is better by {improvement:.1f}% PPL")
            else:
                print(f"  → H2O-L2 is better by {-improvement:.1f}% PPL")
    
    else:
        # Single test with H2O-Attention
        print("\nRunning H2O-Attention evaluation...")
        
        result = evaluate_with_attention_compression(
            model, tokenizer, text,
            start_size=4,
            heavy_hitter_size=64,
            recent_size=444,
            max_tokens=args.max_tokens,
            skip_layers=skip_layers,
            device=device,
        )
        
        print("\n" + "="*60)
        print("H2O-Attention Results")
        print("="*60)
        print(f"  Perplexity:  {result['perplexity']:.2f}")
        print(f"  Accuracy:    {result['accuracy']:.2%}")
        print(f"  Throughput:  {result['throughput']:.2f} tokens/sec")
        print(f"  TTFT:        {result['ttft']:.4f} sec")
        print(f"  TPOT:        {result['tpot']:.4f} sec")
        print(f"  Cache Size:  {result['final_cache_size']} tokens")
    
    print("\nTest completed!")


if __name__ == "__main__":
    main()


