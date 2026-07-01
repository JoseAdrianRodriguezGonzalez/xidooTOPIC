import igraph as ig
from modeling.clustering import LeidenClusterer
def test_leiden_with_weights():

    g = ig.Graph(edges=[(0,1),(1,2),(2,3)])
    g.es["weight"] = [0.9, 0.8, 0.95]

    clusterer = LeidenClusterer()
    labels = clusterer.fit_predict(g)

    assert len(labels) == 4
