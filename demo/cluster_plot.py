###
## cluster_maker: demo for cluster analysis
## James Foadi - University of Bath
## November 2025
##
## This script produces clustering for a group of points in 2D,
## using k-means for k = 2, 3, 4, 5. The input file is the csv
## file 'demo_data.csv' in folder 'data/'.
###

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import run_clustering

OUTPUT_DIR = "demo_output"


def print_section_header(title: str):
    """Helper to print clean headers."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def main(args: List[str]) -> None:
    # --- 1. Introductory Message (User Interaction Mark) ---
    print_section_header("MA52109: Cluster Analysis Demo")
    print("This script will load data, perform K-Means clustering for")
    print("k = 2, 3, 4, and 5, and save the resulting plots and metrics.")
    print("-" * 60)

    if len(args) != 1:
        print("Usage: python demo/cluster_plot.py <input_csv>")
        sys.exit(1)

    input_path = args[0]
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    print(f"Input File: {input_path}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # The CSV is assumed to have two or more data columns
    df = pd.read_csv(input_path)
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(numeric_cols) < 2:
        print("Error: The input CSV must have at least two numeric columns.")
        sys.exit(1)
    feature_cols = numeric_cols[:2]  # Use the first two numeric columns

    # For naming outputs
    base = os.path.splitext(os.path.basename(input_path))[0]

    # --- Feature Explanation ---
    print("\nFeature Analysis:")
    print(f"  Features selected: {', '.join(feature_cols)}")
    print("  Note: These represent spatial coordinates (e.g., X and Y).")
    print("  The algorithm uses these to calculate Euclidean distances.")

    # Keep track of saved files for the final summary
    saved_files = []
    metrics_summary = []

    print_section_header("Processing Clustering (k=2, 3, 4, 5)")
    print("Algorithm: K-Means (sklearn implementation for robustness)")
    print("-" * 60)

    for k in (2, 3, 4, 5):
        print(f"  > Running K-Means with k = {k}...", end=" ")

        # FIX 1: Removed min(k, 3) AND switched to sklearn_kmeans
        output_csv = os.path.join(OUTPUT_DIR, f"{base}_clustered_k{k}.csv")
        result = run_clustering(
            input_path=input_path,
            feature_cols=feature_cols,
            algorithm="sklearn_kmeans",  # Switched to robust implementation
            k=k,
            standardise=True,
            output_path=output_csv,
            random_state=42,
            compute_elbow=False,
        )
        saved_files.append(output_csv)

        # Save cluster plot
        plot_path = os.path.join(OUTPUT_DIR, f"{base}_k{k}.png")
        
        # Improve Title
        fig = result["fig_cluster"]
        ax = fig.axes[0]
        ax.set_title(f"Cluster Separation (k={k})")
        
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        saved_files.append(plot_path)

        # Collect metrics
        metrics = {"k": k}
        metrics.update(result.get("metrics", {}))
        metrics_summary.append(metrics)
        print("Done.")

    # Summarise metrics across k
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_csv = os.path.join(OUTPUT_DIR, f"{base}_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    saved_files.append(metrics_csv)

    # Plot some statistics
    if "silhouette" in metrics_df.columns:
        plt.figure()
        plt.bar(metrics_df["k"], metrics_df["silhouette"], color='purple', alpha=0.7)
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score Comparison")
        stats_path = os.path.join(OUTPUT_DIR, f"{base}_silhouette.png")
        plt.savefig(stats_path, dpi=150)
        plt.close()
        saved_files.append(stats_path)

    # --- 3. Print Summary Table (Enhancing User Interaction) ---
    print_section_header("Final Metrics Summary")
    print("Metric Guide:")
    print(" * Inertia: Compactness of clusters (Lower is better).")
    print(" * Silhouette: Separation distance between clusters (Higher is better).")
    print("-" * 60)
    print(metrics_df.to_string(index=False))
    print("-" * 60)
    
    # Identify best k and Add GENERIC CAUTION NOTE
    if "silhouette" in metrics_df.columns:
        best_run = metrics_df.loc[metrics_df["silhouette"].idxmax()]
        best_k = int(best_run['k'])
        print(f"Automated Suggestion: k = {best_k} has the highest Silhouette Score.")
        
        print("\nINTERPRETATION NOTE:")
        print("  The Silhouette Score measures how similar an object is to its own cluster")
        print("  compared to other clusters. It is a useful guide but not perfect.")
        print("  It may sometimes favour fewer, larger clusters over smaller, distinct ones")
        print("  if the distinct groups are spatially close.")
        print("  Recommendation: Always visually verify the result using the saved plots.")
    
    # --- 4. Explicit Output Summary ---
    print_section_header("Output Manifest")
    print("The following files have been saved:")
    for filepath in saved_files:
        print(f"  - {filepath}")
    print("================================================================")


if __name__ == "__main__":
    main(sys.argv[1:])