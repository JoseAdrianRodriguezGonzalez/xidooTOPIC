import umap
class UMAPReducer:
    def __init__(
        self,
        n_neighbors=30,
        n_components=5,
        metric="cosine",
        random_state=42
    ):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.random_state = random_state
    def fit(self, X):
        self.model_ = umap.UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            metric=self.metric,
            random_state=self.random_state
        )
        self.model_.fit(X)
        self.n_features_in_=X.shape[1]
        return self

    def transform(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("Debes llamar a fit() antes de transform()")
        return self.model_.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.model_.fit_transform(X)
