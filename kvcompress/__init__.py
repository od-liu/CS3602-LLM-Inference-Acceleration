"""
KVCompress: A Unified Library for KV Cache Compression

This library provides multiple KV cache compression strategies for 
efficient LLM inference:

1. L2 Compress (KnormPress): Ratio-based compression keeping low-norm tokens
2. Fix-Size L2: Fixed-size cache with L2-based eviction strategies  
3. StreamingLLM: Attention sinks + sliding window for infinite-length inputs

Usage:
    from kvcompress import l2_compress, streaming_llm_compress, evaluate_with_compression

    # Compress KV cache with L2 method
    compressed = l2_compress(past_key_values, keep_ratio=0.8)

    # Compress with StreamingLLM
    compressed = streaming_llm_compress(past_key_values, start_size=4, recent_size=508)

    # Evaluate with any compression method
    results = evaluate_with_compression(
        model, tokenizer, text,
        compress_fn=l2_compress,
        compress_kwargs={"keep_ratio": 0.8}
    )

References:
    - KnormPress: "A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression"
    - StreamingLLM: "Efficient Streaming Language Models with Attention Sinks" (ICLR 2024)
"""

# Import compression methods
from .methods import (
    # Individual compression functions
    l2_compress,
    fix_size_l2_compress,
    streaming_llm_compress,
    # Method registry
    get_compress_fn,
    list_methods,
    register_method,
    COMPRESS_METHODS,
)

# Import evaluation functions
from .evaluate import (
    evaluate_with_compression,
    evaluate_baseline,
    compare_methods,
)

# Import benchmark functions
from .benchmark import (
    benchmark,
    measure_generation_metrics,
    run_benchmark_suite,
    print_benchmark_summary,
)

# Import utilities
from .utils import (
    to_dynamic_cache,
    normalize_kv_cache,
    get_cache_size_mb,
    get_cache_info,
    get_seq_len,
)


__all__ = [
    # Compression methods
    'l2_compress',
    'fix_size_l2_compress',
    'streaming_llm_compress',
    # Method registry
    'get_compress_fn',
    'list_methods',
    'register_method',
    'COMPRESS_METHODS',
    # Evaluation
    'evaluate_with_compression',
    'evaluate_baseline',
    'compare_methods',
    # Benchmark
    'benchmark',
    'measure_generation_metrics',
    'run_benchmark_suite',
    'print_benchmark_summary',
    # Utilities
    'to_dynamic_cache',
    'normalize_kv_cache',
    'get_cache_size_mb',
    'get_cache_info',
    'get_seq_len',
]

__version__ = '2.0.0'

