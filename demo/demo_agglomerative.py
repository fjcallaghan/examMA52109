###
## cluster_maker: Demo for Agglomerative Clustering
##
## Task 5: Demonstrate the new agglomerative module on 'difficult_dataset.csv'.
###

from __future__ import annotations

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Import the NEW module
try:
    from cluster_maker import fit_agglomerative, select_features
except ImportError:
    from cluster_maker.agglomerative import fit_agglomerative
    from cluster_maker.preprocessing import select_features

INPUT_FILE = os.path.join("data", "difficult_dataset.csv")
OUTPUT_DIR = "agglomerative_demo_output"
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

def print_section_header(title: str):
    """Helper to print clean headers."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def main():
    print_section_header("MA52109: Agglomerative Clustering Demo")
    print("Goal: Attempt to separate concentric rings in 'difficult' data.")
    print("Data: 'difficult_dataset.csv' (Non-convex geometry)")
    print("-" * 60)
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        sys.exit(1)
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- Step 1: Load Data ---
    print("Step 1: Preparing Data...")
    df = pd.read_csv(INPUT_FILE)
    feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:2]
    X_df = select_features(df, feature_cols)
    
    # Using Raw Data to preserve geometry (Standardisation distorts rings)
    X = X_df.to_numpy()
    print(f"  Data loaded: {len(df)} rows.")
    print("  Note: Data kept in raw scale to preserve ring geometry.")
    
    # --- Step 2: Compare Algorithms ---
    n_clusters = 4
    linkages = ["ward", "single"]
    
    print_section_header(f"Step 2: Comparing Linkage Methods (k={n_clusters})")
    print("Method Guide:")
    print(" * Ward:   Minimizes variance. Prefers round, compact blobs.")
    print(" * Single: Nearest-neighbour. Prefers connected, snake-like shapes.")
    print("-" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Agglomerative Clustering Comparison (k={n_clusters})", fontsize=14)
    
    for i, linkage in enumerate(linkages):
        print(f"  > Running '{linkage}' linkage...", end=" ")
        
        labels, _ = fit_agglomerative(X, n_clusters=n_clusters, linkage=linkage)
        
        ax = axes[i]
        for label_id in range(n_clusters):
            mask = labels == label_id
            color = COLORS[label_id % len(COLORS)]
            ax.scatter(X[mask, 0], X[mask, 1], c=color, label=f"Cluster {label_id+1}",
                       alpha=0.6, s=20)
            
        ax.set_title(f"Linkage: '{linkage}'")
        ax.legend(loc="upper right", fontsize="small")
        print("Done.")

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "agglomerative_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    # --- Step 3: Analysis ---
    print_section_header("Step 3: Analysis & Interpretation")
    print(f"Plot saved to: {plot_path}")
    print("\n1. Ward Linkage (Left Plot):")
    print("   * Outcome: FAILED.")
    print("   * Observation: Slices the rings into pie sectors.")
    print("   * Reason: Ward's method assumes clusters are spherical blobs,")
    print("     so it cannot detect the elongated ring structure.")
    
    print("\n2. Single Linkage (Right Plot):")
    print("   * Outcome: FAILED (Chaining Effect).")
    print("   * Observation: Merges rings together incorrectly.")
    print("   * Reason: Although Single Linkage is designed for rings, the data")
    print("     points here are positioned too close together. The algorithm finds")
    print("     'bridges' of noise between rings and merges them.")
    
    print("-" * 60)
    print("CONCLUSION: Standard agglomerative methods struggle with this")
    print("specific dataset density, highlighting the need for density-based")
    print("approaches (like DBSCAN) or Spectral Clustering for such topology.")
    print("============================================================")

if __name__ == "__main__":
    main()