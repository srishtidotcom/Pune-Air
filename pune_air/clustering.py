from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform

try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    BACKEND = "scikit-learn"
except ImportError:
    # Keeps the audit runnable in offline environments; normal installation uses sklearn.
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.cluster.vq import kmeans2
    BACKEND = "scipy fallback"


def labels_for(x: np.ndarray, method: str, k: int, seed: int = 42) -> np.ndarray:
    if BACKEND == "scikit-learn":
        if method == "kmeans":
            return KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(x)
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x)
    if method == "kmeans":
        best_labels, best_sse = None, np.inf
        for trial in range(20):
            try:
                centers, labels = kmeans2(x, k, minit="++", iter=100, seed=seed + trial)
            except Exception:
                continue
            sse = ((x - centers[labels]) ** 2).sum()
            if len(set(labels)) == k and sse < best_sse:
                best_labels, best_sse = labels, sse
        if best_labels is None:
            raise RuntimeError("K-means could not find all clusters")
        return best_labels
    return fcluster(linkage(x, method="ward"), k, criterion="maxclust") - 1


def silhouette(x: np.ndarray, labels: np.ndarray, seed: int = 42) -> float:
    if len(np.unique(labels)) < 2:
        return np.nan
    if BACKEND == "scikit-learn":
        return float(silhouette_score(x, labels, sample_size=min(len(x), 1000), random_state=seed))
    rng = np.random.default_rng(seed)
    ix = rng.choice(len(x), min(len(x), 1000), replace=False)
    xs, ls = x[ix], labels[ix]
    d = squareform(pdist(xs))
    scores = []
    for i in range(len(ix)):
        same = ls == ls[i]
        if same.sum() <= 1:
            scores.append(0.0)
            continue
        a = d[i, same].sum() / (same.sum() - 1)
        b = min(d[i, ls == other].mean() for other in np.unique(ls) if other != ls[i])
        scores.append((b - a) / max(a, b) if max(a, b) else 0)
    return float(np.mean(scores))


def compare_models(x: np.ndarray, ks=range(2, 6)) -> tuple[pd.DataFrame, np.ndarray, dict]:
    rows, models = [], {}
    for method in ("kmeans", "agglomerative"):
        for k in ks:
            labels = labels_for(x, method, k)
            sizes = np.bincount(labels, minlength=k)
            score = silhouette(x, labels)
            viable = bool(sizes.min() >= max(10, int(0.03 * len(x))))
            rows.append({"method": method, "k": k, "silhouette": score,
                         "smallest_cluster": int(sizes.min()), "largest_cluster": int(sizes.max()),
                         "viable": viable})
            models[(method, k)] = labels
    table = pd.DataFrame(rows)
    pool = table[table.viable] if table.viable.any() else table
    winner = pool.sort_values(["silhouette", "k", "method"], ascending=[False, True, True]).iloc[0]
    key = (winner.method, int(winner.k))
    return table, models[key], {"method": key[0], "k": key[1], "silhouette": float(winner.silhouette)}


def name_clusters(x_absolute: np.ndarray, labels: np.ndarray, mode: str) -> pd.DataFrame:
    records = []
    means = x_absolute.mean(axis=1)
    overall = np.median(means)
    for label in sorted(np.unique(labels)):
        curve = np.median(x_absolute[labels == label], axis=0)
        low = float(curve.min()); high = float(curve.max()); peak = int(curve.argmax())
        spread = high - low
        if spread < max(5, 0.15 * float(curve.mean())):
            shape = "fairly even"
        elif peak < 6:
            shape = "overnight high"
        elif peak < 11:
            shape = "morning high"
        elif peak < 17:
            shape = "afternoon high"
        else:
            shape = "evening high"
        level = "higher" if float(np.median(means[labels == label])) >= overall else "lower"
        name = f"{shape}, {level} level" if mode == "absolute" else shape
        records.append({"cluster": int(label), "name": name,
                        "eligible_days": int(sum(labels == label)), "median_daily_mean": float(np.median(means[labels == label])),
                        "peak_hour": peak, "curve_min": low, "curve_max": high})
    return pd.DataFrame(records)
