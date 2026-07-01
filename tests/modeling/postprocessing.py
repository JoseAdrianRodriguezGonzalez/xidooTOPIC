import numpy as np 
from modeling.postprocessing import ClusterFilter,ClusterMerger
def test_cluster_filter_basic():
    emb = np.random.rand(10, 5)
    texts = [f"doc{i}" for i in range(10)]
    labels = np.array([0,0,0,1,1,2,2,2,2,2])  # cluster 1 es pequeño

    cf = ClusterFilter(min_cluster_size=3, verbose=False)

    emb_f, txt_f, lab_f, removed, stats = cf.fit_transform(emb, texts, labels)

    assert 1 in removed
    assert len(lab_f) < len(labels)
    assert all(l != 1 for l in lab_f)
def test_cluster_filter_reindex():
    emb = np.random.rand(6, 5)
    texts = [f"d{i}" for i in range(6)]
    labels = np.array([0,0,5,5,9,9])

    cf = ClusterFilter(min_cluster_size=2, reindex=True, verbose=False)
    _, _, lab_f, _, _ = cf.fit_transform(emb, texts, labels)

    assert set(lab_f) == {0,1,2}
def test_cluster_merger_basic():
    emb = np.vstack([
        np.random.normal(0, 0.1, (10, 5)),
        np.random.normal(0.05, 0.1, (10, 5)),  # muy cercanos → deberían fusionarse
    ])

    labels = np.array([0]*10 + [1]*10)

    merger = ClusterMerger(verbose=False)
    new_labels = merger.fit(emb, labels)

    assert len(np.unique(new_labels)) <= 2
def test_cluster_merger_stable():
    emb = np.random.rand(20, 5)
    labels = np.array([i for i in range(20)])  # todos distintos

    merger = ClusterMerger(max_iter=10)
    new_labels = merger.fit(emb, labels)

    assert len(new_labels) == len(labels)
