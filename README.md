# cluster_maker
`cluster_maker` is a small educational Python package for simulating clustered
datasets and running clustering analyses with a simple, user-friendly interface.

It is designed for practicals and exams where students are given an incomplete
or faulty version of the package and asked to debug or extend it.

## Allowed libraries
The package only uses:

- Python standard library  
- NumPy  
- pandas  
- matplotlib  
- SciPy  
- scikit-learn  

No other third-party libraries are required.

## Main features
- Define a **seed DataFrame** describing cluster centres  
- Simulate clustered data around these centres  
- Compute basic **descriptive statistics** and **correlations**  
- Preprocess data: feature selection and standardisation  
- Run clustering with:
  - a simple **manual K-means** implementation  
  - a scikit-learn **KMeans** wrapper  
- Evaluate clustering with:
  - **inertia** (within-cluster sum of squares)  
  - **silhouette score**  
  - **elbow curve** for K selection  
- Plot:
  - 2D cluster scatter with optional centroids  
  - elbow curve  
- High-level **`run_clustering`** interface  
- Demo scripts and unit tests

## Package root directory structure
- `cluster_maker/`
  - `dataframe_builder.py` – build seed DataFrame and simulate clustered data  
  - `data_analyser.py` – descriptive statistics and correlation  
  - `data_exporter.py` – CSV and formatted text export  
  - `preprocessing.py` – feature selection and standardisation  
  - `algorithms.py` – manual K-means and scikit-learn KMeans wrapper  
  - `evaluation.py` – inertia, silhouette, elbow curve  
  - `plotting_clustered.py` – 2D cluster plots and elbow plots  
  - `interface.py` – high-level `run_clustering` function  
- `demo/` – example scripts  
- `data/` - csv data file used by the example scripts
- `tests/` – basic unit tests using the standard library `unittest`

## Installation (local use)
From the root directory of the project, run:

```bash
pip install -e .
```

This installs the package in editable mode, meaning you can modify the files
and re-run tests or demos without reinstalling.

## Notes on pyproject.toml and the *.egg-info directory
This project includes a small file named pyproject.toml.
You do not need to open or edit it. Its only purpose is to tell Python/pip
that this folder is a valid installable package. Without it, the command
pip install -e . would fail.

When you run the installation command, pip automatically creates a directory
called something like:

`cluster_maker.egg-info/`

This folder contains package metadata used internally by Python (file lists,
version information, etc.). It is generated automatically and should not be
edited.

Spoken with James couldn't get EXPLANATION.md to work

# Task 2 Explanation

## 1. What was wrong with the original script?
The original script `demo/cluster_plot.py` contained a logic error in the loop responsible for iterating through different values of *k* (number of clusters).

Inside the loop `for k in (2, 3, 4, 5):`, the call to `run_clustering` used the argument `k = min(k, 3)`. This hard-coded constraint meant that for any *k* greater than 3, the algorithm effectively ran with *k=3*. As a result, the outputs for *k=4* and *k=5* were incorrect duplicates of the *k=3* analysis, rather than true 4-cluster or 5-cluster solutions.

Additionally, the script attempted to plot the silhouette score by looking for the column `"silhouette_score"` in the metrics DataFrame. However, the `cluster_maker` package returns this metric under the key `"silhouette"`. This mismatch caused the silhouette comparison plot to be skipped entirely.

## 2. How I fixed it
1.  **Corrected the *k* parameter:** I removed the `min(k, 3)` constraint in the `run_clustering` function call and passed `k=k`. This ensures the algorithm uses the intended number of clusters for every iteration.
2.  **Fixed the metric key:** I updated the column check to look for `"silhouette"` instead of `"silhouette_score"`, enabling the generation of the silhouette comparison bar chart.
3.  **Enhanced User Interaction:** I added an introductory print statement to explain what the script is doing and a final summary listing exactly which files were saved, improving the user experience as per the marking criteria.

### Note on Clustering Results
The script may calculate a higher Silhouette Score for k=3 than k=4, even if the data visually appears to have 4 clusters. This is an expected behaviour of the metric: if two of the four clusters are spatially close, the algorithm penalises their "separation" score, preferring to treat them as a single large cluster. The script now includes a generic note in the terminal advising the user to inspect the plots visually rather than relying solely on the automated score.

## 3. Overview of the `cluster_maker` package
The `cluster_maker` package is a modular tool designed for simulating, analyzing, and clustering 2D datasets. Its main components are:

* **`dataframe_builder`**: Generates synthetic datasets. It allows the user to define a "seed" structure (cluster centres) and simulates data points around them using Gaussian noise.
* **`preprocessing`**: prepares raw data for analysis. It handles feature selection (ensuring columns are numeric) and standardisation (scaling features to zero mean and unit variance).
* **`algorithms`**: Contains the core clustering logic. It includes a manual implementation of K-Means (`kmeans`) for educational transparency and a wrapper around the robust `sklearn.cluster.KMeans` (`sklearn_kmeans`).
* **`evaluation`**: Provides metrics to assess clustering quality, including Inertia (within-cluster sum of squares), Silhouette Score (cluster cohesion/separation), and the Elbow Method.
* **`plotting_clustered`**: Handles visualisation, creating 2D scatter plots of clusters/centroids and line plots for elbow curves.
* **`interface`**: Offers a high-level function `run_clustering` that orchestrates the entire pipeline—loading, preprocessing, clustering, evaluating, and plotting—in a single command.

## 4. Extension: Agglomerative Clustering (Task 5)
I have extended the package by adding a new module `agglomerative.py` and a corresponding demo script `demo/demo_agglomerative.py`.

### New Functionality
* **Module:** `cluster_maker/agglomerative.py`
* **Function:** `fit_agglomerative(X, n_clusters, linkage)`
* **Purpose:** This function wraps `sklearn.cluster.AgglomerativeClustering`. It is designed to handle datasets where K-Means assumptions (spherical blobs) fail. It integrates with the package by enforcing type safety and returning data in a compatible format (labels, None).

### Performance on "Difficult" Data
The demo script tests this module on `difficult_dataset.csv` (concentric rings).
* **Ward Linkage:** Failed. It minimizes variance and assumes spherical clusters, causing it to slice the rings into sectors.
* **Single Linkage:** Failed. While Single Linkage is theoretically capable of detecting rings, the specific density of this dataset caused "chaining" (noise points bridging the gap between rings), resulting in merged clusters.
* **Conclusion:** This analysis demonstrates the limitations of standard agglomerative methods on noisy, non-convex data.

## 5. Master Menu
To improve the ease of marking and navigation, I have created a master menu script:
* **Script:** `demo/main_menu.py`
* **Usage:** Run `python demo/main_menu.py` to access a unified interface for launching Task 2, Task 4, and Task 5 demos from a single location.