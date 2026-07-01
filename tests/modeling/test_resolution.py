import numpy as np
import igraph as ig
from modeling.clustering import LeidenResolutionOptimizer
def test_optimizer_basic():
    # Embeddings simples (3 clusters claros)
    emb = np.vstack([
        np.random.normal(0, 0.1, (10, 5)),
        np.random.normal(5, 0.1, (10, 5)),
        np.random.normal(-5, 0.1, (10, 5)),
    ])

    # Grafo simple completamente conectado
    n = len(emb)
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    g = ig.Graph(edges=edges)

    optimizer = LeidenResolutionOptimizer(verbose=True,min_avg_size=1)

    optimizer.fit(g, emb)

    assert optimizer.best_labels_ is not None
    assert optimizer.best_resolution_ is not None
    assert len(optimizer.best_labels_) == n
def test_optimizer_history():
    emb = np.random.rand(20, 5)
    g = ig.Graph.Full(20)

    optimizer = LeidenResolutionOptimizer(min_avg_size=1)

    optimizer.fit(g, emb)

    assert len(optimizer.history_) > 0

    # cada entrada debe tener 5 valores
    for entry in optimizer.history_:
        assert len(entry) == 5
import pytest

def test_optimizer_no_solution():
    emb = np.random.rand(10, 5)

    # Grafo vacío → Leiden no encuentra clusters útiles
    g = ig.Graph(n=10)

    optimizer = LeidenResolutionOptimizer(
        min_avg_size=1000  # imposible
    )

    with pytest.raises(ValueError):
        optimizer.fit(g, emb)
def test_optimizer_reproducibility():
    emb = np.random.rand(30, 5)
    g = ig.Graph.Full(30)

    opt1 = LeidenResolutionOptimizer(seed=42)
    opt2 = LeidenResolutionOptimizer(seed=42)

    labels1 = opt1.fit_predict(g, emb)
    labels2 = opt2.fit_predict(g, emb)

    assert np.array_equal(labels1, labels2)
def test_optimizer_with_weights():
    emb = np.random.rand(20, 5)

    g = ig.Graph.Full(20)
    g.es["weight"] = np.random.rand(g.ecount())

    optimizer = LeidenResolutionOptimizer(min_avg_size=1)

    optimizer.fit(g, emb)

    assert optimizer.best_labels_ is not None
