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


def compute_cluster_cohesion(embeddings, labels):
    """
    Calcula:
    - centroides
    - radio promedio intra-cluster
    """
    centroids = {}
    cohesion = {}

    unique_labels = np.unique(labels)

    for lab in unique_labels:
        cluster_emb = embeddings[labels == lab]
        centroid = cluster_emb.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid)+1e-9)

        centroids[lab] = centroid

        sims = cosine_similarity(cluster_emb, centroid.reshape(1, -1))
        cohesion[lab] = 1 - sims.mean()  # menor = más compacto

    return centroids, cohesion

def compute_global_threshold(sim_matrix, percentile=95):
    values = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    return np.percentile(values, percentile)


def hybrid_threshold_alpha(centroids, cohesion, percentile=95, alpha=0.5):
    labels = list(centroids.keys())
    centroid_matrix = np.array([centroids[l] for l in labels])
    sim_matrix = cosine_similarity(centroid_matrix)

    T_global = compute_global_threshold(sim_matrix, percentile)

    merges = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            lab_i = labels[i]
            lab_j = labels[j]

            sim = sim_matrix[i, j]

            T_local = 1 - (cohesion[lab_i] + cohesion[lab_j]) / 2
            T_dynamic = alpha * T_global + (1 - alpha) * T_local

            if sim > T_dynamic:
                merges.append((lab_i, lab_j, sim, T_dynamic))

    return merges
def composite_cluster_score(
        embeddings,
        labels,
        min_cluster_size=5,
        alpha=0.5,
        beta=0.4,
        gamma=0.1
    ):

    unique_clusters = np.unique(labels)
    N = len(labels)

    # ------------------------
    # 1️⃣ Cohesión intra
    # ------------------------
    intra_scores = []
    weights = []

    for cluster in unique_clusters:
        idx = np.where(labels == cluster)[0]
        cluster_emb = embeddings[idx]

        if len(cluster_emb) < 2:
            continue

        sim_matrix = cosine_similarity(cluster_emb)
        upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        cohesion = upper.mean()

        intra_scores.append(cohesion)
        weights.append(len(idx) / N)

    C_intra = np.sum(np.array(intra_scores) * np.array(weights))

    # ------------------------
    # 2️⃣ Separación inter
    # ------------------------
    centroids = []

    for cluster in unique_clusters:
        idx = np.where(labels == cluster)[0]
        cluster_emb = embeddings[idx]
        centroid = cluster_emb.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids.append(centroid)

    centroids = np.array(centroids)

    if len(centroids) > 1:
        centroid_sim = cosine_similarity(centroids)
        upper = centroid_sim[np.triu_indices_from(centroid_sim, k=1)]
        S_inter = 1 - upper.mean()
    else:
        S_inter = 0

    # ------------------------
    # 3️⃣ Penalización
    # ------------------------
    small_clusters = sum(
        1 for cluster in unique_clusters
        if np.sum(labels == cluster) < min_cluster_size
    )

    P = 1 - (small_clusters / len(unique_clusters))

    # ------------------------
    # Score final
    # ------------------------
    score = alpha * C_intra + beta * S_inter + gamma * P

    return {
        "C_intra": C_intra,
        "S_inter": S_inter,
        "Penalty": P,
        "Composite Score": score
    }

