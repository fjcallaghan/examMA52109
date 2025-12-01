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


def main(args: List[str]) -> None:
    # --- 1. Introductory Message (User Interaction Mark) ---
    print("----------------------------------------------------------------")
    print("MA52109: Cluster Analysis Demo")
    print("----------------------------------------------------------------")
    print("This script will load data, perform K-Means clustering for")
    print("k = 2, 3, 4, and 5, and save the resulting plots and metrics.")
    print("----------------------------------------------------------------\n")

    if len(args) != 1:
        print("Usage: python demo/cluster_plot.py <input_csv>")
        sys.exit(1)

    input_path = args[0]
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

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

    # Keep track of saved files for the final summary
    saved_files = []

    # Main job: run clustering for k = 2, 3, 4, 5
    metrics_summary = []

    for k in (2, 3, 4, 5):
        print(f"Processing k = {k}...")

        # FIX 1: Removed min(k, 3) to allow k=4 and k=5 to run as intended
        output_csv = os.path.join(OUTPUT_DIR, f"{base}_clustered_k{k}.csv")
        result = run_clustering(
            input_path=input_path,
            feature_cols=feature_cols,
            algorithm="kmeans",
            k=k,
            standardise=True,
            output_path=output_csv,
            random_state=42,
            compute_elbow=False,
        )
        saved_files.append(output_csv)

        # Save cluster plot
        plot_path = os.path.join(OUTPUT_DIR, f"{base}_k{k}.png")
        result["fig_cluster"].savefig(plot_path, dpi=150)
        plt.close(result["fig_cluster"])
        saved_files.append(plot_path)

        # Collect metrics
        metrics = {"k": k}
        metrics.update(result.get("metrics", {}))
        metrics_summary.append(metrics)

    # Summarise metrics across k
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_csv = os.path.join(OUTPUT_DIR, f"{base}_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    saved_files.append(metrics_csv)

    # Plot some statistics
    # FIX 2: Check for 'silhouette' (the correct key), not 'silhouette_score'
    if "silhouette" in metrics_df.columns:
        plt.figure()
        plt.bar(metrics_df["k"], metrics_df["silhouette"])
        plt.xlabel("k")
        plt.ylabel("Silhouette score")
        plt.title("Silhouette score for different k")
        stats_path = os.path.join(OUTPUT_DIR, f"{base}_silhouette.png")
        plt.savefig(stats_path, dpi=150)
        plt.close()
        saved_files.append(stats_path)

    # --- 3. Print Summary Table (Enhancing User Interaction) ---
    print("\n----------------------------------------------------------------")
    print("Final Metrics Summary:")
    print("----------------------------------------------------------------")
    print(metrics_df.to_string(index=False))
    print("----------------------------------------------------------------")

    # --- 2. Explicit Output Summary (User Interaction Mark) ---
    print("\nDemo completed successfully.")
    print("Files saved:")
    for filepath in saved_files:
        print(f"  - {filepath}")


if __name__ == "__main__":
    main(sys.argv[1:])