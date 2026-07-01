from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def compute_intra_coherence(emb, labels):
    """
    Promedio de similitud coseno entre cada punto y el centroide de su cluster.
    """

    unique_labels = np.unique(labels)
    total_score = 0
    valid_clusters = 0

    for label in unique_labels:
        cluster_points = emb[labels == label]

        # Ignorar clusters de tamaño 1
        if len(cluster_points) < 2:
            continue

        centroid = np.mean(cluster_points, axis=0, keepdims=True)

        sims = cosine_similarity(cluster_points, centroid)
        cluster_score = np.mean(sims)

        total_score += cluster_score
        valid_clusters += 1

    if valid_clusters == 0:
        return 0

    return total_score / valid_clusters

#Obtiene la coherencia inter cluster
def compute_inter_separation(emb, labels):
    """
    Separación promedio entre centroides de clusters.
    """

    unique_labels = np.unique(labels)
    centroids = []

    for label in unique_labels:
        cluster_points = emb[labels == label]

        if len(cluster_points) < 2:
            continue

        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)

    if len(centroids) < 2:
        return 0

    centroids = np.array(centroids)

    sim_matrix = cosine_similarity(centroids)

    # Tomar solo parte superior sin diagonal
    K = len(centroids)
    sim_sum = 0
    count = 0

    for i in range(K):
        for j in range(i + 1, K):
            sim_sum += sim_matrix[i, j]
            count += 1

    avg_sim = sim_sum / count

    # Convertimos similitud en separación
    separation = 1 - avg_sim

    return separation

