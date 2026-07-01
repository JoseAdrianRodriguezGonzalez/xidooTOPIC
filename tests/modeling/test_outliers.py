import numpy as np 
from modeling.outliers import IsolationForestOutlierRemover
def test_outlier_removal():
    X = np.random.rand(50, 5)
    texts = [f"text {i}" for i in range(50)]

    remover = IsolationForestOutlierRemover(contamination=0.1)
    result = remover.fit_transform(X, texts)

    assert "embeddings" in result
    assert "texts" in result
    assert len(result["embeddings"]) <= 50
