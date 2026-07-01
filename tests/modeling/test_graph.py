import numpy as np
from modeling.graph import NeighborEstimator



def test_neighbor_estimator_runs():
    X = np.random.rand(50, 10)

    estimator = NeighborEstimator(
        k_clusters=3,
        neighbor_range=range(3, 6),  
        n_bootstrap=2,               
        sample_fraction=0.5,
        random_state=42
    )

    estimator.fit(X)

    assert hasattr(estimator, "best_n_neighbors_")
    assert hasattr(estimator, "results_")
def test_best_k_in_range():
    X = np.random.rand(40, 8)

    estimator = NeighborEstimator(
        k_clusters=3,
        neighbor_range=range(3, 7),
        n_bootstrap=2,
        random_state=42
    )

    estimator.fit(X)

    assert estimator.best_n_neighbors_ in range(3, 7)
def test_reproducibility():
    X = np.random.rand(50, 10)

    est1 = NeighborEstimator(
        k_clusters=3,
        neighbor_range=range(3, 6),
        n_bootstrap=2,
        random_state=42
    )

    est2 = NeighborEstimator(
        k_clusters=3,
        neighbor_range=range(3, 6),
        n_bootstrap=2,
        random_state=42
    )

    est1.fit(X)
    est2.fit(X)

    assert est1.best_n_neighbors_ == est2.best_n_neighbors_
def test_results_shape():
    X = np.random.rand(30, 5)

    estimator = NeighborEstimator(
        k_clusters=2,
        neighbor_range=range(3, 6),
        n_bootstrap=1
    )

    estimator.fit(X)

    results = estimator.results_

    assert len(results) == 3  # 3 valores de k
    assert len(results[0]) == 3  # (k, components, stability
def test_small_dataset():
    X = np.random.rand(10, 4)

    estimator = NeighborEstimator(
        k_clusters=2,
        neighbor_range=range(2, 4),
        n_bootstrap=1
    )

    estimator.fit(X)

    assert estimator.best_n_neighbors_ >= 2
