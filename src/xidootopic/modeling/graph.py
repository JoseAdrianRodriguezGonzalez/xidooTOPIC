import random
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from scipy.sparse.csgraph import connected_components
class NeighborEstimator:
    def __init__(self,k_clusters:int,
                 neighbor_range:range=range(3,31),
                 n_bootstrap:int=20,
                 sample_fraction:float=0.8,
                 random_state:int=42,
                 verbose=False) :

        self.k_clusters=k_clusters
        self.neighbor_range=neighbor_range
        self.n_bootstrap=n_bootstrap
        self.sample_fraction=sample_fraction
        self.random_state=random_state
        self.verbose=verbose 
    def fit(self,embeddings):
        best,results=estimate_optimal_neighbors(embeddings,self.k_clusters,self.neighbor_range,self.n_bootstrap,self.sample_fraction,self.random_state,self.verbose)
        
        self.best_n_neighbors_=int(best[0])
        self.results_=results 
        return self
def estimate_optimal_neighbors(
    embeddings,
    k_clusters,
    neighbor_range=range(3, 31),
    n_bootstrap=20,
    sample_fraction=0.8,
    random_state=42,
    verbose:bool=False
):

    rng = np.random.RandomState(random_state)
    results = []

    for n_neighbors in neighbor_range:
        if verbose:
            print(f"\rProbando {n_neighbors} vecinos", end="")

        # 1️⃣ Construir grafo
        A = kneighbors_graph(
            embeddings,
            n_neighbors=n_neighbors,
            mode='connectivity',
            metric='cosine',
            include_self=False
        )

        A = 0.5 * (A + A.T)

        # 2️⃣ Número de componentes conectados
        n_components, _ = connected_components(A)

        # 3️⃣ Estabilidad estructural
        base_model = SpectralClustering(
            n_clusters=k_clusters,
            affinity='nearest_neighbors',
            n_neighbors=n_neighbors,
            random_state=random_state
        )
        base_labels = base_model.fit_predict(embeddings)

        stability_scores = []

        for _ in range(n_bootstrap):

            sample_idx = rng.choice(
                len(embeddings),
                size=int(sample_fraction * len(embeddings)),
                replace=False
            )

            X_sample = embeddings[sample_idx]

            model = SpectralClustering(
                n_clusters=k_clusters,
                affinity='nearest_neighbors',
                n_neighbors=n_neighbors,
                random_state=rng.randint(0, 10000)
            )

            sample_labels = model.fit_predict(X_sample)
            base_subset = base_labels[sample_idx]

            ari = adjusted_rand_score(base_subset, sample_labels)
            stability_scores.append(ari)

        mean_stability = np.mean(stability_scores)

        results.append((n_neighbors, n_components, mean_stability))

    # Convertir a array
    results = np.array(results, dtype=object)

    # Filtrar grafos conectados
    connected = results[results[:, 1] == 1]

    if len(connected) == 0:
        if verbose:        
            print("⚠ Ningún k produjo grafo completamente conectado.")
        best = results[np.argmax(results[:, 2])]
    else:
        # Elegir el menor k con estabilidad cercana al máximo
        max_stab = np.max(connected[:, 2])
        threshold = max_stab * 0.95
        candidates = connected[connected[:, 2] >= threshold]
        best = candidates[np.argmin(candidates[:, 0])]

    return best, results
import faiss
import igraph as ig
from sklearn.preprocessing import normalize
class KNNGraphBuilder:
    def __init__(self,k:int=15,metrics:str="cosine",verbose:bool=False):
        self.k=k
        self.metrics=metrics
        self.verbose=verbose
    def fit(self,X:np.ndarray):
        self.n_samples_,self.n_features_=X.shape
        return self 
    def transform(self,X:np.ndarray):
        if not hasattr(self,"n_samples_"):
            raise RuntimeError("Shoud be called  fit() before transform")
        n,dim=X.shape 
        #  Normalizar embeddings para coseno correcto
        if self.metrics=="cosine":
            X=normalize(X,norm="l2").astype("float32")

        #Índice FAISS para producto punto (equivalente a coseno si normalizado)
        index = faiss.IndexFlatIP(dim)
        index.add(X)
        # Buscar k vecinos (+1 porque el primero es el propio punto)
        similarities, indices = index.search(X, self.k + 1)
        #Inicializa el grafo
        graph = ig.Graph()
        graph.add_vertices(n)

        edges = set()
        weights = []
        #recorre el grafo 
        for i in range(n):
            for j, sim in zip(indices[i][1:], similarities[i][1:]):  # excluir self-loop
                edge = tuple(sorted((i, j)))  # evitar duplicados
                if edge not in edges:
                    edges.add(edge)
                    weights.append(float(sim))
        
        graph.add_edges(list(edges))
        graph.es["weight"] = weights
        if self.verbose:            
            print(f"Grafo construido con {graph.vcount()} nodos y {graph.ecount()} aristas.")
            print(f"Grado promedio ≈ {2*graph.ecount()/graph.vcount():.2f}")
        self.graph_=graph
        return graph
    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)
