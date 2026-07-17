import numpy as np 
from sklearn.ensemble import IsolationForest
class IsolationForestOutlierRemover:
    def __init__(self,contamination:float=0.02,
                 random_state:int=42,
                 n_jobs:int=-1) :
        self.contamination=contamination
        self.random_state=random_state 
        self.n_jobs=n_jobs 
    def fit(self,X:np.ndarray):
        self.model_= IsolationForest(
        contamination=self.contamination,
         random_state=self.random_state,
        n_jobs=self.n_jobs)
        self.model_.fit(X)
        return self
    def transform(self,X:np.ndarray,text:list[str]|None=None):

        if not hasattr(self, "model_"):
            raise RuntimeError("Debes llamar a fit() antes de transform()")
        preds = self.model_.predict(X)
        scores = self.model_.decision_function(X)
        mask = preds == 1
        filtered_embeddings = X[mask]
        result={
            "embeddings":filtered_embeddings,
            "mask":mask,
            "scores":scores,
            "removed_indices":np.where (preds==-1)[0].tolist()
        }
        if text is not None:
            if len(text)!=len(X):
                raise ValueError("texts y X deben tener la misma longitud")
            result["texts"]=[text[i] for i in range(len(text)) if mask[i]]

        self.stats_ = {
            "original_count": len(X),
            "filtered_count": len(filtered_embeddings),
            "removed_count": len(result["removed_indices"]),
            "removed_percentage": round(len(result["removed_indices"]) / len(X) * 100, 2)
        }
        return result
    def fit_transform(self,X:np.ndarray,text:None|list[str]=None):
        self.fit(X)
        return self.transform(X,text)


