from enum import unique
import numpy as np 
class ClusterFilter:
    def __init__(self,min_cluster_size:int=5,reindex:bool=True,verbose:bool=True) :
        self.min_cluster_size=min_cluster_size
        self.reindex=reindex
        self.verbose=verbose 
    def fit(self,labels:np.ndarray):
        unique_clusters,counts=np.unique(labels,return_counts=True)
        self.cluster_sizes=dict(zip(unique_clusters,counts))
        self.valid_clusters = {
            cid for cid, size in self.cluster_sizes.items()
            if size >= self.min_cluster_size
        }
        
        self.removed_cluster_ = [
            cid for cid in unique_clusters
                if cid not in self.valid_clusters
            ]
        return self 
    def transform(self,embeddings:np.ndarray,texts:list[str],labels:np.ndarray):
        mask = np.array([c in self.valid_clusters for c in labels])

        filtered_embeddings = embeddings[mask]
        filtered_texts = [texts[i] for i in range(len(texts)) if mask[i]]
        filtered_cluster_labels = labels[mask]

        stats = {
            "original_clusters": len(self.cluster_sizes),
            "remaining_clusters": len(self.valid_clusters),
            "removed_clusters": len(self.removed_cluster_),
            "removed_documents": int(np.sum(~mask)),
            "remaining_documents": int(np.sum(mask))
        }
        if self.verbose: 
            print("Cluster filtering stats:")
            for k, v in stats.items():
                print(f"{k}: {v}")
        if self.reindex:
            unique = np.unique(filtered_cluster_labels)
            remap = {old: new for new, old in enumerate(unique)}
            filtered_cluster_labels = np.array([remap[c] for c in filtered_cluster_labels])
        return (
            filtered_embeddings,
            filtered_texts,
            filtered_cluster_labels,
            self.removed_cluster_,
            stats
        )
    def fit_transform(self, embeddings, texts, labels):
        self.fit(labels)
        return self.transform(embeddings, texts, labels)


from sklearn.metrics.pairwise import cosine_similarity
from modeling.metrics import compute_cluster_cohesion,hybrid_threshold_alpha
class ClusterMerger:
    def __init__(self,percentile:int=95,alpha:float=0.6,max_iter:int=50,verbose:bool=False,reindex:bool=False):
        self.reindex=reindex
        self.percentile=percentile
        self.alpha=alpha
        self.max_iter=max_iter 
        self.verbose=verbose
    def fit(self,embeddings:np.ndarray,labels:np.ndarray):
        labels_current = labels.copy()
        for iteration in range(self.max_iter):
            
            # 1️⃣ Recalcular centroides y cohesión
            centroids, cohesion = compute_cluster_cohesion(
                embeddings,
                labels_current
            )
            # 2️⃣ Calcular lista de fusiones candidatas
            merges = hybrid_threshold_alpha(
                centroids,
                cohesion,
                percentile=self.percentile,
                alpha=self.alpha 
            )

            if not merges:
                if self.verbose:
                    print(f"Convirgio en {iteration} iteraciones2")
                break
            # 3️⃣ Tomar la mejor fusión (mayor similitud)
            merges_sorted = sorted(merges, key=lambda x: x[2], reverse=True)
            lab_i, lab_j, sim, tau = merges_sorted[0]
            if self.verbose:
                 print(f"merge {lab_j} → {lab_i} (sim={sim:.3f})")
            labels_current[labels_current == lab_j] = lab_i
        if self.reindex:
            unique = np.unique(labels_current)
            remap = {old: new for new, old in enumerate(unique)}
            labels_current = np.array([remap[c] for c in labels_current])  
        return labels_current

