#!/usr/bin/env python3
"""
Benchmark Results Visualization Script

This script generates charts for KV Cache compression benchmark results.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.figsize'] = (12, 6)

# Output directory
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Color palette - modern and professional
COLORS = {
    'baseline': '#2C3E50',      # Dark blue-gray
    'l2': '#E74C3C',            # Red
    'fix_size': '#3498DB',      # Blue
    'streaming': '#27AE60',     # Green
    'recent_only': '#95A5A6',   # Gray
}


def plot_compare_all():
    """Plot comparison of all methods from compare_all benchmark."""
    # Data from compare_all benchmark
    methods = ['Baseline', 'L2\n(kr=0.8)', 'L2\n(kr=0.5)', 'Fix-512\nkeep_low', 'Streaming\n512', 'Streaming\n1024']
    throughput = [82.97, 114.70, 115.93, 79.46, 92.97, 84.26]
    ppl = [15.48, 57.83, 43.74, 17.66, 15.92, 15.52]
    accuracy = [47.77, 30.72, 35.29, 46.40, 47.57, 47.72]
    cache_size = [1999, 99, 99, 512, 512, 1024]
    
    colors = [COLORS['baseline'], COLORS['l2'], COLORS['l2'], 
              COLORS['fix_size'], COLORS['streaming'], COLORS['streaming']]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('KV Cache Compression Methods Comparison (Pythia-2.8B)', fontsize=16, fontweight='bold')
    
    # Throughput
    ax1 = axes[0, 0]
    bars1 = ax1.bar(methods, throughput, color=colors, edgecolor='white', linewidth=1.5)
    ax1.axhline(y=throughput[0], color=COLORS['baseline'], linestyle='--', alpha=0.7, label='Baseline')
    ax1.set_ylabel('Throughput (tokens/sec)')
    ax1.set_title('Throughput ↑')
    for bar, val in zip(bars1, throughput):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}', 
                ha='center', va='bottom', fontsize=9)
    
    # PPL
    ax2 = axes[0, 1]
    bars2 = ax2.bar(methods, ppl, color=colors, edgecolor='white', linewidth=1.5)
    ax2.axhline(y=ppl[0], color=COLORS['baseline'], linestyle='--', alpha=0.7, label='Baseline')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Perplexity ↓')
    for bar, val in zip(bars2, ppl):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', 
                ha='center', va='bottom', fontsize=9)
    
    # Accuracy
    ax3 = axes[1, 0]
    bars3 = ax3.bar(methods, accuracy, color=colors, edgecolor='white', linewidth=1.5)
    ax3.axhline(y=accuracy[0], color=COLORS['baseline'], linestyle='--', alpha=0.7, label='Baseline')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Accuracy ↑')
    ax3.set_ylim(0, 60)
    for bar, val in zip(bars3, accuracy):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', 
                ha='center', va='bottom', fontsize=9)
    
    # Cache Size
    ax4 = axes[1, 1]
    bars4 = ax4.bar(methods, cache_size, color=colors, edgecolor='white', linewidth=1.5)
    ax4.axhline(y=cache_size[0], color=COLORS['baseline'], linestyle='--', alpha=0.7, label='Baseline')
    ax4.set_ylabel('Cache Size (tokens)')
    ax4.set_title('Cache Size ↓')
    for bar, val in zip(bars4, cache_size):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'{val}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'compare_all_methods.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: compare_all_methods.png")


def plot_fix_size_l2():
    """Plot fix_size_l2 benchmark results."""
    # Data from fix_size_l2 benchmark
    methods = ['Baseline', 'Recent\n256', 'Recent\n512', 
               'Fix256\nkr=0.8', 'Fix256\nkr=0.5', 'Fix256\nkr=0.3',
               'Fix512\nkr=0.8', 'Fix512\nkr=0.5', 'Fix512\nkr=0.3']
    
    throughput = [90.10, 107.29, 99.71, 87.16, 87.32, 85.27, 84.73, 79.81, 79.51]
    ppl = [15.48, 48.68, 32.75, 20.47, 19.86, 19.76, 17.96, 17.66, 17.83]
    accuracy = [47.77, 35.49, 39.27, 44.15, 44.82, 45.05, 46.08, 46.40, 46.10]
    cache_size = [1999, 256, 512, 256, 256, 256, 512, 512, 512]
    
    # Color assignment
    colors = [COLORS['baseline']] + [COLORS['recent_only']]*2 + [COLORS['fix_size']]*6
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Fix-Size L2 Compression Benchmark (Pythia-2.8B)', fontsize=16, fontweight='bold')
    
    # Throughput
    ax1 = axes[0, 0]
    bars1 = ax1.bar(methods, throughput, color=colors, edgecolor='white', linewidth=1.5)
    ax1.axhline(y=throughput[0], color=COLORS['baseline'], linestyle='--', alpha=0.7)
    ax1.set_ylabel('Throughput (tokens/sec)')
    ax1.set_title('Throughput ↑ (higher is better)')
    ax1.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars1, throughput):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', 
                ha='center', va='bottom', fontsize=8)
    
    # PPL
    ax2 = axes[0, 1]
    bars2 = ax2.bar(methods, ppl, color=colors, edgecolor='white', linewidth=1.5)
    ax2.axhline(y=ppl[0], color=COLORS['baseline'], linestyle='--', alpha=0.7)
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Perplexity ↓ (lower is better)')
    ax2.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars2, ppl):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}', 
                ha='center', va='bottom', fontsize=8)
    
    # Accuracy
    ax3 = axes[1, 0]
    bars3 = ax3.bar(methods, accuracy, color=colors, edgecolor='white', linewidth=1.5)
    ax3.axhline(y=accuracy[0], color=COLORS['baseline'], linestyle='--', alpha=0.7)
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Accuracy ↑ (higher is better)')
    ax3.set_ylim(0, 60)
    ax3.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars3, accuracy):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%', 
                ha='center', va='bottom', fontsize=8)
    
    # Improvement vs Recent-Only (scatter plot)
    ax4 = axes[1, 1]
    # Compare Fix-Size vs Recent-Only for same cache sizes
    categories = ['256 tokens', '512 tokens']
    recent_ppl = [48.68, 32.75]
    fix_best_ppl = [19.76, 17.66]  # Best fix-size PPL for each cache size
    
    x = np.arange(len(categories))
    width = 0.35
    bars_recent = ax4.bar(x - width/2, recent_ppl, width, label='Recent-Only', color=COLORS['recent_only'], edgecolor='white')
    bars_fix = ax4.bar(x + width/2, fix_best_ppl, width, label='Fix-Size L2 (best)', color=COLORS['fix_size'], edgecolor='white')
    ax4.axhline(y=15.48, color=COLORS['baseline'], linestyle='--', alpha=0.7, label='Baseline (PPL=15.48)')
    ax4.set_ylabel('Perplexity')
    ax4.set_title('Fix-Size L2 vs Recent-Only (PPL Comparison)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    
    # Add improvement percentage
    for i, (r, f) in enumerate(zip(recent_ppl, fix_best_ppl)):
        improvement = (r - f) / r * 100
        ax4.text(i, max(r, f) + 1, f'↓{improvement:.0f}%', ha='center', fontsize=10, fontweight='bold', color=COLORS['fix_size'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fix_size_l2_benchmark.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fix_size_l2_benchmark.png")


def plot_streaming_llm():
    """Plot StreamingLLM benchmark results."""
    # Data from streaming_llm benchmark
    methods = ['Baseline', 'Recent\n256', 'Recent\n512', 'Recent\n1024',
               'Streaming\n256', 'Streaming\n512', 'Streaming\n1024']
    
    throughput = [83.97, 106.82, 99.28, 88.76, 102.49, 96.06, 84.69]
    ppl = [15.48, 48.68, 32.75, 20.91, 16.61, 15.92, 15.52]
    accuracy = [47.77, 35.49, 39.27, 43.67, 47.07, 47.57, 47.72]
    cache_size = [1999, 256, 512, 1024, 256, 512, 1024]
    
    colors = [COLORS['baseline']] + [COLORS['recent_only']]*3 + [COLORS['streaming']]*3
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('StreamingLLM Benchmark (Pythia-2.8B)', fontsize=16, fontweight='bold')
    
    # Throughput
    ax1 = axes[0, 0]
    bars1 = ax1.bar(methods, throughput, color=colors, edgecolor='white', linewidth=1.5)
    ax1.axhline(y=throughput[0], color=COLORS['baseline'], linestyle='--', alpha=0.7)
    ax1.set_ylabel('Throughput (tokens/sec)')
    ax1.set_title('Throughput ↑ (higher is better)')
    for bar, val in zip(bars1, throughput):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', 
                ha='center', va='bottom', fontsize=9)
    
    # PPL
    ax2 = axes[0, 1]
    bars2 = ax2.bar(methods, ppl, color=colors, edgecolor='white', linewidth=1.5)
    ax2.axhline(y=ppl[0], color=COLORS['baseline'], linestyle='--', alpha=0.7)
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Perplexity ↓ (lower is better)')
    for bar, val in zip(bars2, ppl):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}', 
                ha='center', va='bottom', fontsize=9)
    
    # Accuracy
    ax3 = axes[1, 0]
    bars3 = ax3.bar(methods, accuracy, color=colors, edgecolor='white', linewidth=1.5)
    ax3.axhline(y=accuracy[0], color=COLORS['baseline'], linestyle='--', alpha=0.7)
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Accuracy ↑ (higher is better)')
    ax3.set_ylim(0, 60)
    for bar, val in zip(bars3, accuracy):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%', 
                ha='center', va='bottom', fontsize=9)
    
    # StreamingLLM vs Recent-Only comparison
    ax4 = axes[1, 1]
    cache_sizes = ['256', '512', '1024']
    streaming_ppl = [16.61, 15.92, 15.52]
    recent_ppl = [48.68, 32.75, 20.91]
    
    x = np.arange(len(cache_sizes))
    width = 0.35
    bars_recent = ax4.bar(x - width/2, recent_ppl, width, label='Recent-Only', color=COLORS['recent_only'], edgecolor='white')
    bars_streaming = ax4.bar(x + width/2, streaming_ppl, width, label='StreamingLLM', color=COLORS['streaming'], edgecolor='white')
    ax4.axhline(y=15.48, color=COLORS['baseline'], linestyle='--', alpha=0.7, label='Baseline (PPL=15.48)')
    ax4.set_ylabel('Perplexity')
    ax4.set_title('StreamingLLM vs Recent-Only (PPL Comparison)')
    ax4.set_xlabel('Cache Size (tokens)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(cache_sizes)
    ax4.legend()
    
    # Add improvement percentage
    for i, (r, s) in enumerate(zip(recent_ppl, streaming_ppl)):
        improvement = (r - s) / r * 100
        ax4.text(i, max(r, s) + 1.5, f'↓{improvement:.0f}%', ha='center', fontsize=10, fontweight='bold', color=COLORS['streaming'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'streaming_llm_benchmark.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: streaming_llm_benchmark.png")


def plot_tradeoff_analysis():
    """Plot PPL vs Throughput tradeoff analysis."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # All methods data
    data = {
        'Baseline': {'throughput': 82.97, 'ppl': 15.48, 'acc': 47.77, 'color': COLORS['baseline'], 'marker': 'o', 'size': 200},
        'L2 kr=0.8': {'throughput': 114.70, 'ppl': 57.83, 'acc': 30.72, 'color': COLORS['l2'], 'marker': '^', 'size': 150},
        'L2 kr=0.5': {'throughput': 115.93, 'ppl': 43.74, 'acc': 35.29, 'color': COLORS['l2'], 'marker': '^', 'size': 150},
        'Fix256 kr=0.3': {'throughput': 85.27, 'ppl': 19.76, 'acc': 45.05, 'color': COLORS['fix_size'], 'marker': 's', 'size': 120},
        'Fix512 kr=0.5': {'throughput': 79.81, 'ppl': 17.66, 'acc': 46.40, 'color': COLORS['fix_size'], 'marker': 's', 'size': 140},
        'Streaming 256': {'throughput': 102.49, 'ppl': 16.61, 'acc': 47.07, 'color': COLORS['streaming'], 'marker': 'D', 'size': 150},
        'Streaming 512': {'throughput': 96.06, 'ppl': 15.92, 'acc': 47.57, 'color': COLORS['streaming'], 'marker': 'D', 'size': 180},
        'Streaming 1024': {'throughput': 84.69, 'ppl': 15.52, 'acc': 47.72, 'color': COLORS['streaming'], 'marker': 'D', 'size': 200},
    }
    
    for name, d in data.items():
        ax.scatter(d['throughput'], d['ppl'], c=d['color'], s=d['size'], 
                   marker=d['marker'], label=name, edgecolors='white', linewidth=2, alpha=0.9)
    
    # Add annotations
    for name, d in data.items():
        offset = (5, 5) if 'L2' not in name else (5, -15)
        ax.annotate(name, (d['throughput'], d['ppl']), textcoords="offset points", 
                   xytext=offset, fontsize=9, alpha=0.8)
    
    # Highlight optimal region (high throughput, low PPL)
    ax.axhline(y=20, color='green', linestyle=':', alpha=0.5)
    ax.axvline(x=90, color='green', linestyle=':', alpha=0.5)
    ax.fill_between([90, 130], 0, 20, alpha=0.1, color='green', label='Optimal Region')
    
    ax.set_xlabel('Throughput (tokens/sec) ↑', fontsize=12)
    ax.set_ylabel('Perplexity ↓', fontsize=12)
    ax.set_title('Throughput vs Perplexity Tradeoff (Pythia-2.8B)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(70, 125)
    ax.set_ylim(10, 65)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'tradeoff_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: tradeoff_analysis.png")


def plot_summary_table():
    """Create a summary visualization."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Table data
    headers = ['Method', 'Cache Size', 'Throughput', 'PPL', 'Accuracy', 'Recommendation']
    data = [
        ['Baseline', '1999', '82.97', '15.48', '47.77%', 'Quality Baseline'],
        ['StreamingLLM-256', '256', '102.49 (+23%)', '16.61 (+7%)', '47.07%', 'BEST: Speed'],
        ['StreamingLLM-512', '512', '96.06 (+16%)', '15.92 (+3%)', '47.57%', 'BEST: Balanced'],
        ['StreamingLLM-1024', '1024', '84.69 (+2%)', '15.52 (+0.3%)', '47.72%', 'BEST: Quality'],
        ['Fix512 keep_low', '512', '79.81 (-4%)', '17.66 (+14%)', '46.40%', 'Alternative'],
        ['L2 kr=0.5', '99', '115.93 (+40%)', '43.74 (+183%)', '35.29%', 'WARN: High PPL'],
    ]
    
    # Create table
    table = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Style recommendation cells
    for i in range(1, len(data) + 1):
        rec_col = len(headers) - 1  # Last column index
        if 'BEST' in data[i-1][-1]:
            table[(i, rec_col)].set_facecolor('#E8F8F5')
        elif 'WARN' in data[i-1][-1]:
            table[(i, rec_col)].set_facecolor('#FDEDEC')
    
    ax.set_title('KV Cache Compression Methods Summary (Pythia-2.8B)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'summary_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: summary_table.png")


def main():
    """Generate all plots."""
    print("Generating benchmark visualization charts...")
    print(f"Output directory: {RESULTS_DIR}")
    print("-" * 50)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    plot_compare_all()
    plot_fix_size_l2()
    plot_streaming_llm()
    plot_tradeoff_analysis()
    plot_summary_table()
    
    print("-" * 50)
    print("All charts generated successfully!")


if __name__ == "__main__":
    main()

