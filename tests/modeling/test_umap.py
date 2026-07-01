from modeling.reduction import UMAPReducer
import numpy as np 
def test_umap_reducer():
    X = np.random.rand(20, 10)

    reducer = UMAPReducer(n_components=3)
    X_red = reducer.fit_transform(X)

    assert X_red.shape == (20, 3)
