#!/usr/bin/env python3
"""
Plot TFLOPS over time comparing TNN framework vs Torch distributed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ResNet50 FLOPs for training (forward + backward)
FLOPS_PER_IMAGE_TRAIN = 12.3e9  # 12.3 GFLOPs per image

def calculate_tflops(flops, time_seconds):
    """Calculate TFLOPS given FLOPs and time in seconds."""
    return flops / time_seconds / 1e12

def load_tnn_data():
    """Load TNN framework data (20260518_134111)."""
    batch_file = Path("logs/tnn_imagenet100_resnet50_batch_20260518_134111.csv")
    df = pd.read_csv(batch_file)
    
    batch_size = 128
    
    # Convert time from ms to seconds
    df['time_sec'] = df['time_ms'] / 1000.0
    
    # Calculate cumulative time (for x-axis)
    df['cumulative_time_sec'] = df['time_sec'].cumsum()
    df['cumulative_time_min'] = df['cumulative_time_sec'] / 60.0
    
    # Calculate FLOPs per batch
    df['flops_per_batch'] = batch_size * FLOPS_PER_IMAGE_TRAIN
    
    # Calculate TFLOPS per batch
    df['tflops'] = df.apply(
        lambda row: calculate_tflops(row['flops_per_batch'], row['time_sec']),
        axis=1
    )
    
    return df

def load_torch_data():
    """Load Torch distributed data (20260518_160631)."""
    metrics_file = Path("logs/resnet50_imagenet100_20260518_160631_rank0_metrics.csv")
    df = pd.read_csv(metrics_file)
    
    # Filter training batches only
    df_train = df[df['phase'] == 'train_batch'].copy()
    
    # Use compute+comm time for TFLOPS calculation
    df_train['compute_time_sec'] = df_train['compute_comm_time']
    
    # Calculate cumulative time
    df_train['cumulative_time_sec'] = df_train['batch_total_time'].cumsum()
    df_train['cumulative_time_min'] = df_train['cumulative_time_sec'] / 60.0
    
    # Calculate FLOPs per batch
    df_train['flops_per_batch'] = df_train['batch_size'] * FLOPS_PER_IMAGE_TRAIN
    
    # Calculate TFLOPS per batch (using compute time)
    df_train['tflops'] = df_train.apply(
        lambda row: calculate_tflops(row['flops_per_batch'], row['compute_time_sec']),
        axis=1
    )
    
    return df_train

def plot_tflops_comparison(tnn_df, torch_df):
    """Create comparison plots."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('TFLOPS Comparison: TNN Framework vs Torch Distributed\nResNet50 Training on ImageNet100', 
                 fontsize=14, fontweight='bold')
    
    # Color scheme
    tnn_color = '#2E7D32'      # Green for TNN
    torch_color = '#D32F2F'     # Red for Torch
    
    # Plot 1: TFLOPS over time (full timeline)
    ax1 = axes[0, 0]
    ax1.plot(tnn_df['cumulative_time_min'], tnn_df['tflops'], 
             label='TNN Framework', color=tnn_color, alpha=0.6, linewidth=1)
    ax1.plot(torch_df['cumulative_time_min'], torch_df['tflops'], 
             label='Torch Distributed', color=torch_color, alpha=0.6, linewidth=1)
    ax1.set_xlabel('Time (minutes)', fontsize=11)
    ax1.set_ylabel('TFLOPS', fontsize=11)
    ax1.set_title('TFLOPS Over Time (Full Training)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add mean lines
    tnn_mean = tnn_df['tflops'].mean()
    torch_mean = torch_df['tflops'].mean()
    ax1.axhline(y=tnn_mean, color=tnn_color, linestyle='--', linewidth=2, 
                label=f'TNN Mean: {tnn_mean:.2f} TFLOPS', alpha=0.8)
    ax1.axhline(y=torch_mean, color=torch_color, linestyle='--', linewidth=2, 
                label=f'Torch Mean: {torch_mean:.2f} TFLOPS', alpha=0.8)
    ax1.legend(fontsize=9, loc='best')
    
    # Plot 2: TFLOPS over time (first 1000 steps for detail)
    ax2 = axes[0, 1]
    tnn_subset = tnn_df.head(1000)
    torch_subset = torch_df.head(1000)
    
    ax2.plot(tnn_subset['step'], tnn_subset['tflops'], 
             label='TNN Framework', color=tnn_color, alpha=0.6, linewidth=1)
    ax2.plot(torch_subset['step'], torch_subset['tflops'], 
             label='Torch Distributed', color=torch_color, alpha=0.6, linewidth=1)
    ax2.set_xlabel('Training Step', fontsize=11)
    ax2.set_ylabel('TFLOPS', fontsize=11)
    ax2.set_title('TFLOPS Over Steps (First 1000 Steps)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rolling average TFLOPS
    ax3 = axes[1, 0]
    window = 50
    tnn_rolling = tnn_df['tflops'].rolling(window=window, center=True).mean()
    torch_rolling = torch_df['tflops'].rolling(window=window, center=True).mean()
    
    ax3.plot(tnn_df['step'], tnn_rolling, 
             label=f'TNN (rolling avg, window={window})', color=tnn_color, linewidth=2)
    ax3.plot(torch_df['step'], torch_rolling, 
             label=f'Torch (rolling avg, window={window})', color=torch_color, linewidth=2)
    ax3.set_xlabel('Training Step', fontsize=11)
    ax3.set_ylabel('TFLOPS (Rolling Average)', fontsize=11)
    ax3.set_title(f'Smoothed TFLOPS ({window}-step Rolling Average)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: TFLOPS distribution (histogram)
    ax4 = axes[1, 1]
    ax4.hist(tnn_df['tflops'], bins=50, alpha=0.6, color=tnn_color, 
             label='TNN Framework', edgecolor='black', linewidth=0.5)
    ax4.hist(torch_df['tflops'], bins=50, alpha=0.6, color=torch_color, 
             label='Torch Distributed', edgecolor='black', linewidth=0.5)
    ax4.axvline(x=tnn_mean, color=tnn_color, linestyle='--', linewidth=2, 
                label=f'TNN Mean: {tnn_mean:.2f}')
    ax4.axvline(x=torch_mean, color=torch_color, linestyle='--', linewidth=2, 
                label=f'Torch Mean: {torch_mean:.2f}')
    ax4.set_xlabel('TFLOPS', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('TFLOPS Distribution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save the figure
    output_file = 'tflops_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_file}")
    
    # Show the plot
    plt.show()
    
    return fig

def print_statistics(tnn_df, torch_df):
    """Print detailed statistics."""
    print("\n" + "=" * 80)
    print("DETAILED STATISTICS")
    print("=" * 80)
    
    print("\nTNN Framework (20260518_134111):")
    print(f"  Total batches:     {len(tnn_df)}")
    print(f"  Mean TFLOPS:       {tnn_df['tflops'].mean():.3f}")
    print(f"  Median TFLOPS:     {tnn_df['tflops'].median():.3f}")
    print(f"  Std Dev:           {tnn_df['tflops'].std():.3f}")
    print(f"  Min TFLOPS:        {tnn_df['tflops'].min():.3f}")
    print(f"  Max TFLOPS:        {tnn_df['tflops'].max():.3f}")
    print(f"  Total time:        {tnn_df['cumulative_time_min'].iloc[-1]:.2f} minutes")
    
    print("\nTorch Distributed (20260518_160631):")
    print(f"  Total batches:     {len(torch_df)}")
    print(f"  Mean TFLOPS:       {torch_df['tflops'].mean():.3f}")
    print(f"  Median TFLOPS:     {torch_df['tflops'].median():.3f}")
    print(f"  Std Dev:           {torch_df['tflops'].std():.3f}")
    print(f"  Min TFLOPS:        {torch_df['tflops'].min():.3f}")
    print(f"  Max TFLOPS:        {torch_df['tflops'].max():.3f}")
    print(f"  Total time:        {torch_df['cumulative_time_min'].iloc[-1]:.2f} minutes")
    
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    tnn_mean = tnn_df['tflops'].mean()
    torch_mean = torch_df['tflops'].mean()
    speedup = tnn_mean / torch_mean
    
    print(f"\nTNN vs Torch:")
    print(f"  TNN Mean TFLOPS:       {tnn_mean:.3f}")
    print(f"  Torch Mean TFLOPS:     {torch_mean:.3f}")
    print(f"  Speedup (TNN/Torch):   {speedup:.2f}x")
    print(f"  Performance gain:      {(speedup - 1) * 100:+.1f}%")
    
    if speedup > 1:
        print(f"\n✓ TNN Framework is {speedup:.2f}x FASTER than Torch Distributed!")
    else:
        print(f"\n✗ Torch Distributed is {1/speedup:.2f}x faster than TNN Framework")

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Loading Training Data...")
    print("=" * 80)
    
    # Load data
    print("\nLoading TNN framework data...")
    tnn_df = load_tnn_data()
    print(f"  ✓ Loaded {len(tnn_df)} batches")
    
    print("\nLoading Torch distributed data...")
    torch_df = load_torch_data()
    print(f"  ✓ Loaded {len(torch_df)} batches")
    
    # Print statistics
    print_statistics(tnn_df, torch_df)
    
    # Create plots
    print("\n" + "=" * 80)
    print("Generating Plots...")
    print("=" * 80)
    
    plot_tflops_comparison(tnn_df, torch_df)
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80 + "\n")
