import random
import numpy as np 
import leidenalg 
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

