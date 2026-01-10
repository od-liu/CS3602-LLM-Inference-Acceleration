#!/usr/bin/env python3
"""
Plot comprehensive benchmark results for all KV cache compression methods.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

# Benchmark data from comprehensive test
data_512 = {
    'baseline': {'ttft': 0.0251, 'tpot': 0.0122, 'throughput': 81.36, 'ppl': 15.48, 'acc': 47.77, 'cache': 1999},
    'recent_only_512': {'ttft': 0.0085, 'tpot': 0.0098, 'throughput': 99.95, 'ppl': 32.75, 'acc': 39.27, 'cache': 512},
    'streaming_512': {'ttft': 0.0084, 'tpot': 0.0102, 'throughput': 96.73, 'ppl': 15.92, 'acc': 47.57, 'cache': 512},
    'h2o_l2_512': {'ttft': 0.0086, 'tpot': 0.0116, 'throughput': 84.53, 'ppl': 15.78, 'acc': 47.57, 'cache': 512},
    'snapkv_512': {'ttft': 0.0086, 'tpot': 0.0127, 'throughput': 77.58, 'ppl': 19.08, 'acc': 45.52, 'cache': 512},
    'pyramid_512': {'ttft': 0.0085, 'tpot': 0.0111, 'throughput': 88.71, 'ppl': 17.37, 'acc': 46.30, 'cache': 414},
    'adaptive_512': {'ttft': 0.0084, 'tpot': 0.0125, 'throughput': 79.15, 'ppl': 19.82, 'acc': 45.25, 'cache': 256},
    'fix_l2_512': {'ttft': 0.0086, 'tpot': 0.0123, 'throughput': 79.95, 'ppl': 17.66, 'acc': 46.40, 'cache': 512},
}

data_1024 = {
    'baseline': {'ttft': 0.0251, 'tpot': 0.0122, 'throughput': 81.36, 'ppl': 15.48, 'acc': 47.77, 'cache': 1999},
    'recent_only_1024': {'ttft': 0.0085, 'tpot': 0.0110, 'throughput': 89.47, 'ppl': 20.91, 'acc': 43.67, 'cache': 1024},
    'streaming_1024': {'ttft': 0.0088, 'tpot': 0.0115, 'throughput': 85.38, 'ppl': 15.52, 'acc': 47.72, 'cache': 1024},
    'h2o_l2_1024': {'ttft': 0.0084, 'tpot': 0.0128, 'throughput': 77.24, 'ppl': 15.53, 'acc': 47.82, 'cache': 1024},
    'snapkv_1024': {'ttft': 0.0085, 'tpot': 0.0136, 'throughput': 72.71, 'ppl': 16.23, 'acc': 47.27, 'cache': 1024},
    'pyramid_1024': {'ttft': 0.0085, 'tpot': 0.0113, 'throughput': 87.06, 'ppl': 16.40, 'acc': 47.17, 'cache': 829},
    'adaptive_1024': {'ttft': 0.0084, 'tpot': 0.0124, 'throughput': 79.35, 'ppl': 17.73, 'acc': 46.22, 'cache': 512},
    'fix_l2_1024': {'ttft': 0.0084, 'tpot': 0.0130, 'throughput': 75.55, 'ppl': 16.17, 'acc': 46.80, 'cache': 1024},
}

def plot_512_comparison():
    """Plot comparison of all methods at 512 cache size."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = ['baseline', 'recent_only_512', 'streaming_512', 'h2o_l2_512', 
               'snapkv_512', 'pyramid_512', 'adaptive_512', 'fix_l2_512']
    labels = ['Baseline', 'Recent-Only', 'StreamingLLM', 'H2O-L2', 
              'SnapKV-Lite', 'Pyramid KV', 'Adaptive L2', 'Fix-L2']
    
    colors = ['#2C3E50', '#E74C3C', '#27AE60', '#3498DB', 
              '#9B59B6', '#F39C12', '#1ABC9C', '#E67E22']
    
    x = np.arange(len(methods))
    
    # Throughput
    ax1 = axes[0, 0]
    throughputs = [data_512[m]['throughput'] for m in methods]
    bars = ax1.bar(x, throughputs, color=colors)
    ax1.set_ylabel('Throughput (tokens/sec)')
    ax1.set_title('Throughput Comparison (512 Cache)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.axhline(y=data_512['baseline']['throughput'], color='red', linestyle='--', alpha=0.5, label='Baseline')
    for i, v in enumerate(throughputs):
        ax1.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    # PPL
    ax2 = axes[0, 1]
    ppls = [data_512[m]['ppl'] for m in methods]
    bars = ax2.bar(x, ppls, color=colors)
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Perplexity Comparison (512 Cache) - Lower is Better')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.axhline(y=data_512['baseline']['ppl'], color='red', linestyle='--', alpha=0.5, label='Baseline')
    for i, v in enumerate(ppls):
        ax2.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Accuracy
    ax3 = axes[1, 0]
    accs = [data_512[m]['acc'] for m in methods]
    bars = ax3.bar(x, accs, color=colors)
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Accuracy Comparison (512 Cache)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45, ha='right')
    ax3.axhline(y=data_512['baseline']['acc'], color='red', linestyle='--', alpha=0.5, label='Baseline')
    for i, v in enumerate(accs):
        ax3.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Cache Size
    ax4 = axes[1, 1]
    caches = [data_512[m]['cache'] for m in methods]
    bars = ax4.bar(x, caches, color=colors)
    ax4.set_ylabel('KV Cache Size (tokens)')
    ax4.set_title('KV Cache Size Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=45, ha='right')
    for i, v in enumerate(caches):
        ax4.text(i, v + 20, f'{v}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/methods_512_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: results/methods_512_comparison.png")
    plt.close()


def plot_1024_comparison():
    """Plot comparison of all methods at 1024 cache size."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = ['baseline', 'recent_only_1024', 'streaming_1024', 'h2o_l2_1024', 
               'snapkv_1024', 'pyramid_1024', 'adaptive_1024', 'fix_l2_1024']
    labels = ['Baseline', 'Recent-Only', 'StreamingLLM', 'H2O-L2', 
              'SnapKV-Lite', 'Pyramid KV', 'Adaptive L2', 'Fix-L2']
    
    colors = ['#2C3E50', '#E74C3C', '#27AE60', '#3498DB', 
              '#9B59B6', '#F39C12', '#1ABC9C', '#E67E22']
    
    x = np.arange(len(methods))
    
    # Throughput
    ax1 = axes[0, 0]
    throughputs = [data_1024[m]['throughput'] for m in methods]
    bars = ax1.bar(x, throughputs, color=colors)
    ax1.set_ylabel('Throughput (tokens/sec)')
    ax1.set_title('Throughput Comparison (1024 Cache)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.axhline(y=data_1024['baseline']['throughput'], color='red', linestyle='--', alpha=0.5, label='Baseline')
    for i, v in enumerate(throughputs):
        ax1.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    # PPL
    ax2 = axes[0, 1]
    ppls = [data_1024[m]['ppl'] for m in methods]
    bars = ax2.bar(x, ppls, color=colors)
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Perplexity Comparison (1024 Cache) - Lower is Better')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.axhline(y=data_1024['baseline']['ppl'], color='red', linestyle='--', alpha=0.5, label='Baseline')
    for i, v in enumerate(ppls):
        ax2.text(i, v + 0.3, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Accuracy
    ax3 = axes[1, 0]
    accs = [data_1024[m]['acc'] for m in methods]
    bars = ax3.bar(x, accs, color=colors)
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Accuracy Comparison (1024 Cache)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45, ha='right')
    ax3.axhline(y=data_1024['baseline']['acc'], color='red', linestyle='--', alpha=0.5, label='Baseline')
    for i, v in enumerate(accs):
        ax3.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Cache Size
    ax4 = axes[1, 1]
    caches = [data_1024[m]['cache'] for m in methods]
    bars = ax4.bar(x, caches, color=colors)
    ax4.set_ylabel('KV Cache Size (tokens)')
    ax4.set_title('KV Cache Size Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=45, ha='right')
    for i, v in enumerate(caches):
        ax4.text(i, v + 20, f'{v}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/methods_1024_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: results/methods_1024_comparison.png")
    plt.close()


def plot_ppl_vs_throughput():
    """Plot PPL vs Throughput tradeoff for all methods."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # All methods data
    all_data = {
        'Baseline': (81.36, 15.48, 1999, '#2C3E50'),
        'Recent-Only 512': (99.95, 32.75, 512, '#E74C3C'),
        'Recent-Only 1024': (89.47, 20.91, 1024, '#C0392B'),
        'StreamingLLM 512': (96.73, 15.92, 512, '#27AE60'),
        'StreamingLLM 1024': (85.38, 15.52, 1024, '#229954'),
        'H2O-L2 512': (84.53, 15.78, 512, '#3498DB'),
        'H2O-L2 1024': (77.24, 15.53, 1024, '#2980B9'),
        'SnapKV 512': (77.58, 19.08, 512, '#9B59B6'),
        'SnapKV 1024': (72.71, 16.23, 1024, '#8E44AD'),
        'Pyramid 512': (88.71, 17.37, 414, '#F39C12'),
        'Pyramid 1024': (87.06, 16.40, 829, '#D68910'),
        'Adaptive 512': (79.15, 19.82, 256, '#1ABC9C'),
        'Adaptive 1024': (79.35, 17.73, 512, '#16A085'),
        'Fix-L2 512': (79.95, 17.66, 512, '#E67E22'),
        'Fix-L2 1024': (75.55, 16.17, 1024, '#CA6F1E'),
    }
    
    for name, (throughput, ppl, cache, color) in all_data.items():
        size = cache / 10  # Scale marker size by cache
        ax.scatter(throughput, ppl, s=size, c=color, alpha=0.7, label=f'{name} ({cache})')
        ax.annotate(name.replace(' ', '\n'), (throughput, ppl), 
                   textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    
    ax.set_xlabel('Throughput (tokens/sec)')
    ax.set_ylabel('Perplexity (lower is better)')
    ax.set_title('PPL vs Throughput Tradeoff\n(Marker size = KV Cache size)')
    
    # Add ideal region annotation
    ax.axhline(y=16, color='green', linestyle='--', alpha=0.3)
    ax.axvline(x=85, color='green', linestyle='--', alpha=0.3)
    ax.annotate('Ideal Region\n(High throughput, Low PPL)', 
               xy=(95, 15.5), fontsize=10, color='green', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('results/ppl_throughput_tradeoff.png', dpi=150, bbox_inches='tight')
    print("Saved: results/ppl_throughput_tradeoff.png")
    plt.close()


def plot_method_summary():
    """Create a summary heatmap of all methods."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate relative performance vs baseline
    baseline = {'throughput': 81.36, 'ppl': 15.48, 'acc': 47.77}
    
    methods = [
        'recent_only_512', 'streaming_512', 'h2o_l2_512', 'snapkv_512', 
        'pyramid_512', 'adaptive_512', 'fix_l2_512',
        'recent_only_1024', 'streaming_1024', 'h2o_l2_1024', 'snapkv_1024',
        'pyramid_1024', 'adaptive_1024', 'fix_l2_1024'
    ]
    
    labels = [
        'Recent-Only 512', 'StreamingLLM 512', 'H2O-L2 512', 'SnapKV 512',
        'Pyramid 512', 'Adaptive 512', 'Fix-L2 512',
        'Recent-Only 1024', 'StreamingLLM 1024', 'H2O-L2 1024', 'SnapKV 1024',
        'Pyramid 1024', 'Adaptive 1024', 'Fix-L2 1024'
    ]
    
    all_data = {**data_512, **data_1024}
    
    # Calculate relative changes (%)
    throughput_change = [(all_data[m]['throughput'] / baseline['throughput'] - 1) * 100 for m in methods]
    ppl_change = [(all_data[m]['ppl'] / baseline['ppl'] - 1) * 100 for m in methods]
    acc_change = [(all_data[m]['acc'] / baseline['acc'] - 1) * 100 for m in methods]
    
    # Create heatmap data
    data_matrix = np.array([throughput_change, ppl_change, acc_change]).T
    
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=-30, vmax=30)
    
    # Set labels
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Throughput %', 'PPL % (↓ better)', 'Accuracy %'])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    
    # Add text annotations
    for i in range(len(labels)):
        for j in range(3):
            val = data_matrix[i, j]
            # Invert color logic for PPL (negative is good)
            if j == 1:  # PPL column
                text_color = 'white' if val > 20 else 'black'
            else:
                text_color = 'white' if abs(val) > 15 else 'black'
            ax.text(j, i, f'{val:+.1f}%', ha='center', va='center', 
                   color=text_color, fontsize=9, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Change from Baseline (%)')
    
    ax.set_title('Method Performance Summary vs Baseline\n(Green = Better for Throughput/Acc, Red = Better for PPL)')
    
    # Add separating line between 512 and 1024
    ax.axhline(y=6.5, color='black', linewidth=2)
    ax.text(-0.7, 3, '512\nCache', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(-0.7, 10, '1024\nCache', ha='center', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/method_summary_heatmap.png', dpi=150, bbox_inches='tight')
    print("Saved: results/method_summary_heatmap.png")
    plt.close()


def plot_cache_efficiency():
    """Plot cache size efficiency (quality per cache token)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    all_data = {**data_512, **data_1024}
    
    methods = ['streaming_512', 'h2o_l2_512', 'snapkv_512', 'pyramid_512', 
               'fix_l2_512', 'streaming_1024', 'h2o_l2_1024', 'snapkv_1024', 
               'pyramid_1024', 'fix_l2_1024']
    
    labels = ['Stream\n512', 'H2O\n512', 'Snap\n512', 'Pyramid\n512', 'Fix\n512',
              'Stream\n1024', 'H2O\n1024', 'Snap\n1024', 'Pyramid\n1024', 'Fix\n1024']
    
    # Quality score = (baseline_ppl / method_ppl) * (method_acc / baseline_acc) * 100
    baseline_ppl = 15.48
    baseline_acc = 47.77
    
    quality_scores = []
    for m in methods:
        d = all_data[m]
        score = (baseline_ppl / d['ppl']) * (d['acc'] / baseline_acc) * 100
        quality_scores.append(score)
    
    cache_sizes = [all_data[m]['cache'] for m in methods]
    efficiency = [q / c * 100 for q, c in zip(quality_scores, cache_sizes)]  # Quality per 100 tokens
    
    colors = ['#27AE60', '#3498DB', '#9B59B6', '#F39C12', '#E67E22'] * 2
    
    x = np.arange(len(methods))
    bars = ax.bar(x, efficiency, color=colors)
    
    ax.set_ylabel('Quality Score per 100 Cache Tokens')
    ax.set_xlabel('Method')
    ax.set_title('Cache Efficiency: Quality per Cache Token\n(Higher = Better efficiency)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    for i, (v, c) in enumerate(zip(efficiency, cache_sizes)):
        ax.text(i, v + 0.5, f'{v:.1f}\n({c})', ha='center', va='bottom', fontsize=8)
    
    # Add separating line
    ax.axvline(x=4.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(2, max(efficiency) * 0.95, '512 Cache', ha='center', fontsize=12, fontweight='bold')
    ax.text(7, max(efficiency) * 0.95, '1024 Cache', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/cache_efficiency.png', dpi=150, bbox_inches='tight')
    print("Saved: results/cache_efficiency.png")
    plt.close()


if __name__ == "__main__":
    print("Generating comprehensive benchmark charts...")
    plot_512_comparison()
    plot_1024_comparison()
    plot_ppl_vs_throughput()
    plot_method_summary()
    plot_cache_efficiency()
    print("\nAll charts generated successfully!")


