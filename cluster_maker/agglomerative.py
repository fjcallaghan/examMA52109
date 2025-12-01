###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering

def fit_agglomerative(
    X: np.ndarray,
    n_clusters: int,
    linkage: str = "ward",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Perform Agglomerative Hierarchical Clustering using scikit-learn.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    n_clusters : int
        The number of clusters to find.
    linkage : {'ward', 'complete', 'average', 'single'}, default 'ward'
        Which linkage criterion to use.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels for each point.
    centroids : None
        Agglomerative clustering does not produce centroids.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    
    if n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer.")

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    
    return labels, None