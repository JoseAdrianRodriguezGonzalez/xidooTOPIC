import random
import numpy as np 
import leidenalg 
from .metrics import compute_intra_coherence,compute_inter_separation
class LeidenClusterer:
    def __init__(self,resolution=1.0,
                 partition_type=leidenalg.RBConfigurationVertexPartition,
                 use_weights:bool=True,
                 random_state:int=42,
                 verbose:bool=True) :
        self.resolution=resolution
        self.partition_type=partition_type
        self.use_weights=use_weights
        self.random_state=random_state
        self.verbose=verbose
    def fit(self,graph):    
        weights=None 
        if self.use_weights:
            if "weight" in graph.es.attributes():
                weights=graph.es["weight"]
            elif self.verbose:
                print("Grafo sin pesos, usando grafo no ponderado")
        partition = leidenalg.find_partition(
            graph,
            self.partition_type,
            weights=weights,
            resolution_parameter=self.resolution,
            seed=self.random_state
        )
        self.labels_=np.array(partition.membership)
        self.n_clusters_=len(np.unique(self.labels_))
        self.partition_=partition
        if self.verbose: 
            print(f"Clusters encontrados: {self.n_clusters_}")

        return self
    def fit_predict(self,graph):
        self.fit(graph)
        return self.labels_
class LeidenResolutionOptimizer:
    def __init__(self,
                 res_min:float=0.4,
                 res_max:float=1.4,
                 step:float=0.05,
                 alpha:float=0.25,
                 min_avg_size:int=5,
                 verbose:bool=False,
                 seed:int=42) :

        self.res_min=res_min
        self.res_max=res_max
        self.step=step
        self.alpha=alpha
        self.min_avg_size=min_avg_size
        self.random_state=seed 
        self.verbose=verbose
    def fit(self,graph,embeddings):
        best_score = -np.inf
        best_res = None
        best_labels = None
        N = len(embeddings)
        resolutions = np.arange(self.res_min, self.res_max + self.step, self.step)
        self.history_=[]
        for res in resolutions:
            if res<=0:
                continue 
            clusterer=LeidenClusterer(resolution=res,use_weights=False,random_state=self.random_state)
            labels=clusterer.fit_predict(graph)
            n_clusters=clusterer.n_clusters_
            if n_clusters<=1:
                continue 
            avg_cluster_size=N/n_clusters 
            if avg_cluster_size<self.min_avg_size:
                continue 
            coh=compute_intra_coherence(embeddings,labels)
            sep=compute_inter_separation(embeddings,labels)
            
            cluster_penalty=self.alpha*(n_clusters/np.sqrt(N))
            score=coh+sep-cluster_penalty
            self.history_.append((res,n_clusters,coh,sep,score))
            if self.verbose:
                print(f"res={res:.2f} | K={n_clusters} | coh={coh:.3f} | sep={sep:.3f} | score={score:.3f}")
            if score > best_score:
                best_score = score
                best_res = res
                best_labels = labels
        if best_res is None:
            raise ValueError("No se encontro una reosluciona optrimiza valida dentro del rango especificado y los criterios.")
        self.best_resolution_=best_res 
        self.best_labels_=best_labels
        self.best_score=best_score 
        return self
    def fit_predict(self, graph, embeddings):
        self.fit(graph, embeddings)
        return self.best_labels_,self.best_resolution_

