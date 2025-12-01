###
## cluster_maker: Simulated Clustering Analysis
## [Your Name/Student ID]
##
## Task 4: Analyse 'simulated_data.csv' to determine a plausible
## separation of the data into clusters.
###

from __future__ import annotations

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Import tools from our package
from cluster_maker import run_clustering, calculate_descriptive_statistics

# Configuration
INPUT_FILE = os.path.join("data", "simulated_data.csv")
OUTPUT_DIR = "simulated_clustering_output"


def print_section_header(title: str):
    """Helper to print clean headers."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def main():
    print_section_header("MA52109: Simulated Data Clustering Analysis")
    print(f"Input Data:      {INPUT_FILE}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("-" * 60)
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Step 1: Exploratory Data Analysis (EDA) ---
    print_section_header("Step 1: Data Inspection & Statistics")
    df = pd.read_csv(INPUT_FILE)
    
    # --- EXPLAINING THE FEATURES (User Interaction) ---
    feature_names = list(df.columns)
    print(f"Features Detected ({len(feature_names)}): {', '.join(feature_names)}")
    print("-" * 60)
    print("What are these features?")
    print("  These columns represent abstract spatial coordinates (dimensions).")
    print("  Think of them like 'X' and 'Y' positions on a map.")
    print("  The clustering algorithm uses these numbers to calculate the")
    print("  distance between points and group nearby ones together.")
    print("-" * 60)

    # Calculate stats
    stats = calculate_descriptive_statistics(df)
    
    # --- CLARIFYING STATISTICS (User Interaction) ---
    display_stats = stats.copy()
    display_stats.rename(index={
        "count": "Count (N)",
        "mean": "Mean (Average)",
        "std": "Standard Deviation",
        "min": "Minimum Value",
        "25%": "25th Percentile (Lower Q)",
        "50%": "Median (Middle Value)",
        "75%": "75th Percentile (Upper Q)",
        "max": "Maximum Value"
    }, inplace=True)
    
    print("\nSummary Statistics (First 5 columns shown):")
    print(display_stats.iloc[:, :5])
    print("-" * 60)
    
    print("Guide to Statistics:")
    print(" * Standard Deviation: A measure of how spread out the numbers are.")
    print(" * Percentiles (25%/75%): The boundary values for the middle 50% of the data.")
    print(" * Median: The exact middle value (often more typical than the Average).")
    
    # Save original stats to CSV
    stats.to_csv(os.path.join(OUTPUT_DIR, "descriptive_statistics.csv"))

    # --- Step 2: Clustering Analysis ---
    print_section_header("Step 2: Clustering Grid Search (k=2 to 8)")
    print("Algorithm: K-Means (sklearn implementation)")
    print("Method:    We will test different cluster counts (k) to find the best fit.")
    print("-" * 60)
    
    metrics_history = []
    k_values = range(2, 9)
    
    for k in k_values:
        print(f"  > Testing separation into k={k} clusters...", end=" ")
        
        output_csv = os.path.join(OUTPUT_DIR, f"clustered_k{k}.csv")
        
        result = run_clustering(
            input_path=INPUT_FILE,
            feature_cols=df.columns, 
            algorithm="sklearn_kmeans", 
            k=k,
            standardise=True,
            output_path=output_csv,
            random_state=42, 
            compute_elbow=False
        )
        
        # --- FIX: Plot Adjustments ---
        fig = result["fig_cluster"]
        ax = fig.axes[0]
        
        # 1. Fix Title Overlap
        ax.set_title(f"Cluster Separation (k={k})")
        
        # 2. Fix Legend Overlap (Move to Bottom Left)
        ax.legend(loc="lower left")
        
        fig.savefig(os.path.join(OUTPUT_DIR, f"plot_k{k}.png"), dpi=150)
        plt.close(fig)
        
        # Collect Metrics
        m = result["metrics"]
        metrics_history.append({
            "k": k,
            "inertia": m.get("inertia"),
            "silhouette": m.get("silhouette")
        })
        print("Done.")

    # --- Step 3: Visualisation & Conclusion ---
    print_section_header("Step 3: Evaluation & Conclusion")
    metrics_df = pd.DataFrame(metrics_history)
    
    # Plot Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(metrics_df["k"], metrics_df["inertia"], 'bo-', linewidth=2)
    ax1.set_title("Elbow Method (Inertia)")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax1.grid(True)
    
    ax2.bar(metrics_df["k"], metrics_df["silhouette"], color='green', alpha=0.7)
    ax2.set_title("Silhouette Score")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Score")
    ax2.grid(axis='y')
    
    comparison_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=150)
    plt.close()
    
    # Find best k
    best_run = metrics_df.loc[metrics_df["silhouette"].idxmax()]
    best_k = int(best_run["k"])
    best_score = best_run["silhouette"]
    
    print("Metric Definitions:")
    print(" * Inertia: Measures how 'tight' the clusters are (Lower is better).")
    print(" * Silhouette Score: Measures how distinct the clusters are (Higher is better).")
    print("-" * 60)
    print(metrics_df.to_string(index=False))
    print("-" * 60)
    
    print(f"CONCLUSION: The analysis suggests the optimal number of clusters is k = {best_k}.")
    print(f"(Reason: Highest Silhouette Score of {best_score:.4f})")
    
    print("\nNote: K-Means uses linear boundaries. Some points near boundaries may")
    print("be assigned to the nearest geometric centroid, even if they appear")
    print("visually closer to a different group.")
    
    print_section_header("Output Summary")
    print(f"1. Comparison plot saved to: {comparison_path}")
    print(f"2. Individual cluster plots: {OUTPUT_DIR}/plot_k*.png")
    print(f"3. Labelled data files:      {OUTPUT_DIR}/clustered_k*.csv")
    print("============================================================")

if __name__ == "__main__":
    main()