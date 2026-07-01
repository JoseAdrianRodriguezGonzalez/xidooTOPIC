

"""#Librerias"""

import os
import json
import math
import pickle
import leidenalg
import unicodedata
import numpy as np
import igraph as ig
import pandas as pd
#from llama_cpp import Llama
from keybert import KeyBERT
from scipy.linalg import eigh
from google.colab import files
from google.colab import drive
from __future__ import annotations
from scipy.sparse.csgraph import laplacian
from sklearn.preprocessing import normalize
from sklearn.ensemble import IsolationForest
from collections import Counter, defaultdict
from typing import Dict, Literal, Optional, List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModel, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM

drive.mount('/content/drive')
#PARA EMBEDINGSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSs
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

#SUMARIZER
# Carga el pipeline con el modelo que elijas
modelSum="Narrativa/bsc_roberta2roberta_shared-spanish-finetuned-mlsum-summarization" # Changed to a summarization model
tok = AutoTokenizer.from_pretrained(modelSum)
model_summarization = AutoModelForSeq2SeqLM.from_pretrained(modelSum)
#KEYBERT
kw_model = KeyBERT(model=MODEL_NAME)  # reutiliza tu modelo de embeddings

if device == "cuda":
    try:
        model = model.to(device)
        model_summarization = model_summarization.to(device)
        model = model.half()  # SOLO en GPU
        print("Running in FP16 on GPU.")
    except RuntimeError as e:
        print(f"Warning: Could not move model to GPU. Running on CPU. Error: {e}")
        device = "cpu"
        model = model.to("cpu")
        model_summarization = model_summarization.to(device)
else:
    model = model.to("cpu")
    model_summarization = model_summarization.to("cpu")
    print("Running in FP32 on CPU.")

prompt = """You are a data analyst specialized in topic modeling evaluation.

Your task is to analyze the output of a topic modeling process based strictly and exclusively on the information provided.

You will receive:
1. A list of keywords extracted for a topic.
2. A set of representative tourist reviews associated with that topic.

IMPORTANT RULES:

1. You must base your analysis ONLY on the keywords and documents provided.
2. You must NOT assume, infer, or introduce any geographical locations, destinations, cities, landmarks, or contextual information that are not explicitly mentioned in the provided text.
3. If the documents mention a location explicitly, you may refer to it. Otherwise, do not introduce any location.
4. If the keywords and documents do not appear semantically coherent or do not share a clear common theme, you must classify the topic as: "INCOHERENT – DISCARD".
5. Do not speculate.
6. Do not add external knowledge.
7. Be concise, analytical, and objective.

For each topic, provide:

- Topic ID:
- Decision: (VALID / INCOHERENT – DISCARD)
- Proposed Topic Label (max 6 words, neutral and descriptive):
- Short Analytical Description (2–3 sentences strictly grounded in the provided text)

After analyzing all topics:

- Identify if any VALID topics are semantically overlapping.
- If so, propose a merged topic label and indicate which Topic IDs should be merged.
- Do not propose to merge topics unless there is strong semantic overlap explicitly visible in the data.
- If you are uncertain, propose to DISCARD rather than inventing a thematic interpretation.

Output must follow this structure exactly.

below are the input information from the topic modeling process:

"""
"""#Funciones"""

def get_topk_frequent(input_list):
    # Count the frequency of each element
    counts = Counter(input_list)

    # Get the k most common elements and their counts as a list of tuples
    # Example output: [(element1, count1), (element2, count2), ...]
    top_k_with_counts = counts.most_common(len(input_list))

    # Extract only the elements into a list
    top_k_elements = [item[0] for item in top_k_with_counts]
    doc = nlp(" ".join(top_k_elements))
    lematizados = [token.lemma_.lower() for token in doc]
    seen = set()
    deduped = []
    for lemma in lematizados:
        if lemma not in seen:
            seen.add(lemma)
            deduped.append(lemma)
    return deduped

#CLUSTERING BASADO EN GRAFOS (LEIDEN) =========================================


#Agrupa textos en clusters
def group_by_cluster(texts, clusters):
    cluster_dict = {}

    for idx, cluster_id in enumerate(clusters):
        cluster_dict.setdefault(cluster_id, []).append(texts[idx])

    return cluster_dict

#ELIMINA CLUSTERES CON MUY POCOS DOCUMENTOS
def filter_small_clusters(embeddings: np.ndarray, texts: list, cluster_labels: np.ndarray, min_cluster_size: int = 5 ):
    """
    Elimina clusters con tamaño menor al mínimo especificado.

    Retorna:
    --------
    filtered_embeddings
    filtered_texts
    filtered_cluster_labels
    removed_cluster_ids
    stats
    """
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = dict(zip(unique_clusters, counts))

    # Identificar clusters válidos
    valid_clusters = {
        cid for cid, size in cluster_sizes.items()
        if size >= min_cluster_size
    }

    mask = np.array([c in valid_clusters for c in cluster_labels])

    filtered_embeddings = embeddings[mask]
    filtered_texts = [texts[i] for i in range(len(texts)) if mask[i]]
    filtered_cluster_labels = cluster_labels[mask]

    removed_cluster_ids = [
        cid for cid in unique_clusters
        if cid not in valid_clusters
    ]

    stats = {
        "original_clusters": len(unique_clusters),
        "remaining_clusters": len(valid_clusters),
        "removed_clusters": len(removed_cluster_ids),
        "removed_documents": int(np.sum(~mask)),
        "remaining_documents": int(np.sum(mask))
    }

    return (
        filtered_embeddings,
        filtered_texts,
        filtered_cluster_labels,
        removed_cluster_ids,
        stats
    )


#SEMANTIC MERGIN OF CENTROIDS =============================================================================================

#CHECA COHERENCIA INTRACLUSTER
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
        centroid = centroid / np.linalg.norm(centroid)

        centroids[lab] = centroid

        sims = cosine_similarity(cluster_emb, centroid.reshape(1, -1))
        cohesion[lab] = 1 - sims.mean()  # menor = más compacto

    return centroids, cohesion



#OBTIENE THRESHOLD GLOBAL
def compute_global_threshold(sim_matrix, percentile=95):
    values = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    return np.percentile(values, percentile)



# THRESHOLD HYBRIDO
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



#FUSIONA CLUSTERS
def merge_clusters_hybrid(embeddings, labels, percentile=95, alpha=0.6):
    """
    Fusión iterativa de clusters usando threshold híbrido dinámico.

    Args:
        embeddings: np.ndarray (n_samples, dim)
        labels: array con etiquetas de Leiden
        percentile: percentil global
        alpha: peso global vs local

    Returns:
        new_labels: etiquetas finales tras fusiones
    """

    labels_current = labels.copy()
    unique_labels = np.unique(labels_current)

    merged = True

    while merged:
        merged = False

        # 1️⃣ Recalcular centroides y cohesión
        centroids, cohesion = compute_cluster_cohesion(
            embeddings,
            labels_current
        )

        # 2️⃣ Calcular lista de fusiones candidatas
        merges = hybrid_threshold_alpha(
            centroids,
            cohesion,
            percentile=percentile,
            alpha=alpha
        )

        if not merges:
            break

        # 3️⃣ Tomar la mejor fusión (mayor similitud)
        merges_sorted = sorted(merges, key=lambda x: x[2], reverse=True)
        lab_i, lab_j, sim, tau = merges_sorted[0]

        # 4️⃣ Fusionar j en i
        labels_current[labels_current == lab_j] = lab_i

        merged = True

    return labels_current


#OBTIENE LAS KEYWORDS DE CADA CENTROIDE
def extract_cluster_keywordsKF(texts, embeddings, labels, top_docs_ratio=0.2, top_n=10):
    cluster_keywords = {}

    for cluster_id in np.unique(labels):

        cluster_mask = labels == cluster_id
        cluster_texts = np.array(texts)[cluster_mask]
        cluster_embs = embeddings[cluster_mask]

        # 1️⃣ Centroide
        centroid = cluster_embs.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        # 2️⃣ Seleccionar documentos más centrales
        sims = cosine_similarity(cluster_embs, centroid.reshape(1, -1)).flatten()
        n_top = max(1, int(len(sims) * top_docs_ratio))
        top_indices = sims.argsort()[-n_top:]

        listadocs =  cluster_texts[top_indices]
        topk = []
        resumen = " "
        for l in listadocs:
          keywords = kw_model.extract_keywords(
            l,
            keyphrase_ngram_range=(1, 1),
            top_n=10,
            vectorizer=CountVectorizer(stop_words=spanish_stopwords),
            use_mmr=True,
            diversity=0.7
          )
          topk = topk + [tupla[0] for tupla in keywords]

        keys = get_topk_frequent(topk)
        if len(keys) > 50:
          resumenFinal = (", ").join(keys[:50])
        else:
          resumenFinal = (",").join(keys)
        resumenFinal = "[KEYWORDS] "+resumenFinal
        # 3️⃣ Extraer resumen
        """inputs = tok(resumenFinal, return_tensors="pt", max_length=512, truncation=True).to(device)
        with torch.inference_mode():
              out = model_summarization.generate(
                  **inputs,
                  num_beams=5,
                  repetition_penalty=1.5,
                  no_repeat_ngram_size=3,
                  early_stopping=True
              )"""
        resumen = resumenFinal #tok.decode(out[0], skip_special_tokens=True)
        cluster_keywords[cluster_id] = (keys[1:10],resumen)
    return cluster_keywords


#Saca kw y las formatea en para el prompt
def procesar_diccionario_topicos(datos):
    lista_keywords = []
    texto_formateado = ""

    for id_topico, (keywords, resumen) in datos.items():
        # 1. Agregamos las keywords a la lista maestra
        aux = keywords #obtener_lista_palabras(keywords)
        lista_keywords.append(aux)

        # 2. Construimos el string formateado para este tópico
        # Usamos ", ".join() para que las keywords se vean limpias
        formato_entrada = (
            f"Topico: {id_topico}\n"
            f"keywords: {', '.join(aux)}\n"
            f"Documento:\n"
            f"{resumen}\n"
            f"{'-'*30}\n" # Separador visual opcional
        )

        texto_formateado += formato_entrada

    return lista_keywords, texto_formateado




#HERRAMIENTAS PARA OBTENER LA RESOLUCIÓN DEL GRAFO

#Obtiene la resolución óptima

#Evalua la coherencia intra cluster

#Obtiene los K documentos más representativos
def get_representative_documents_with_scores(embeddings, labels, documents, top_k=5):

    result = {}
    unique_clusters = np.unique(labels)
    print(f'\nClusters finales: {len(unique_clusters)}\n')
    for cluster_id in unique_clusters:

        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_emb = embeddings[cluster_indices]

        centroid = np.mean(cluster_emb, axis=0).reshape(1, -1)
        centroid = centroid / np.linalg.norm(centroid)

        sims = cosine_similarity(cluster_emb, centroid).flatten()

        sorted_idx = np.argsort(-sims)

        top_indices = cluster_indices[sorted_idx[:top_k]]

        result[cluster_id] = [
            (documents[i], sims[sorted_idx[j]])
            for j, i in enumerate(top_indices)
        ]

    return result


#Obtiene todas las palabras clave del diccionario
def obtener_keyw(diccionario):
    listas_resultado = []

    # Iteramos sobre los ítems del diccionario (clave y su lista de datos)
    # Usamos sorted() por si quieres que el resultado respete el orden numérico de las claves
    for clave in sorted(diccionario.keys()):
        datos = diccionario[clave]

        # Extraemos solo el texto (primer elemento de cada sublista)
        # Python resuelve automáticamente el Unicode (ej: \u00f3 -> ó) al procesar la cadena
        palabras = [item[0] for item in datos]

        # Unimos las palabras con comas y las añadimos a nuestra lista final
        cadena_palabras = ", ".join(palabras)
        listas_resultado.append(cadena_palabras)

    return listas_resultado


#EVALUACIÓN DE LA COHESION INTRA Y SEPARACIÓN DE CLUSTER BASADO EN EMBEDDINGS Y SIM COSENO

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

# AUXILIAR DE LLM ===============================================================

def format_llm_input_string(cluster_keywords_dict, representative_docs_dict):
    strin = ""
    # Get unique sorted cluster IDs that are present in both dictionaries
    cluster_ids = sorted(list(set(cluster_keywords_dict.keys()) & set(representative_docs_dict.keys())))

    for topic_id in cluster_ids:
        # Convert np.int64 to standard int for f-string formatting if necessary
        strin += f"Topic ID: {int(topic_id)}:\n"

        # Get keywords for this topic_id
        # cluster_keywords_dict[topic_id] is a list of tuples: [('keyword', score), ...]
        keywords_list = [item[0] for item in cluster_keywords_dict[topic_id]]
        strin += f"Keywords:\n{', '.join(keywords_list)}\n"

        strin += f"Documents:\n"
        # representative_docs_dict[topic_id] is a list of tuples: [('document_text', score), ...]
        for i, (doc_text, doc_score) in enumerate(representative_docs_dict[topic_id]):
            strin += f"{i+1}: {doc_text}\n"
        strin += "\n" # Add an extra newline for better separation

    return strin


#BM25 MODIFICADO PARA OBTENER KEYWORDS ================================================
def compute_cluster_bm25(
    cluster_texts,
    k1=1.5,
    b=0.75,
    min_df=1
):
    """
    cluster_texts: dict {cluster_id: [lista de documentos (tokens)]}
                   cada documento debe ser lista de tokens
    """

    # Construir documento por cluster
    cluster_docs = {}
    for cid, docs in cluster_texts.items():
        tokens = []
        for doc in docs:
            tokens.extend(doc)
        cluster_docs[cid] = tokens

    N = len(cluster_docs)

    # Longitudes
    doc_lengths = {cid: len(tokens) for cid, tokens in cluster_docs.items()}
    avgdl = np.mean(list(doc_lengths.values()))

    # Document frequency por cluster
    df = defaultdict(int)
    for cid, tokens in cluster_docs.items():
        unique_terms = set(tokens)
        for term in unique_terms:
            df[term] += 1

    # BM25 scoring
    cluster_scores = {}

    for cid, tokens in cluster_docs.items():
        tf = Counter(tokens)
        scores = {}

        for term, freq in tf.items():

            if df[term] < min_df:
                continue

            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)

            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * (doc_lengths[cid] / avgdl))

            score = idf * (numerator / denominator)

            scores[term] = score

        # ordenar por score
        sorted_terms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        cluster_scores[cid] = sorted_terms

    return cluster_scores


def get_tokenized_grouped_texts(texts: list[str], labels: np.ndarray) -> dict:
    """
    Groups texts by their cluster labels and tokenizes each text.

    Args:
        texts (list[str]): A list of cleaned text documents.
        labels (np.ndarray): An array of cluster labels corresponding to the texts.

    Returns:
        dict: A dictionary where keys are cluster IDs and values are lists of
              tokenized documents belonging to that cluster.
    """
    # Group texts by cluster
    grouped_texts = group_by_cluster(texts, labels)

    # Tokenize each text within the grouped clusters
    tokenized_grouped_texts = {}
    for cid, docs_in_cluster in grouped_texts.items():
        tokenized_grouped_texts[cid] = [text.split() for text in docs_in_cluster]

    return tokenized_grouped_texts


def obtener_lista_palabras(lista_tuplas):
    # Extraemos solo el primer elemento (la palabra) de cada tupla en cada lista del diccionario
   return [palabra for palabra, valor in lista_tuplas]

"""#Ejecución sin creación de embeddings"""

columna = 'comment'
colecti = 'trip'
dataset = 'rm-T'
keypat1 = f'/content/drive/MyDrive/papers/verano2026/results/{colecti}/key-{dataset}.csv'
global_path = f'/content/drive/MyDrive/papers/verano2026/data/{colecti}/{dataset}.npy'
global_pathx = f'/content/drive/MyDrive/papers/verano2026/data/{colecti}/{dataset}.csv'
global_path2 = f'/content/drive/MyDrive/papers/verano2026/dataAux/{colecti}/'
model_path = f'/content/drive/MyDrive/papers/verano2026/LLM/mistral-7b-instruct-v0.2.Q4_K_M.gguf'


data = pd.read_csv(global_pathx, encoding='utf-8')
print(f"\nComienza proceso en dataset: {dataset} \n")
#Limpia Reviews
reviews = data[columna].tolist()

valid_idx, reviews_c = clean_reviews(reviews)

#Obtiene chunks
cleaned_reviews = recortaTodo(reviews_c)

# COMIENZA EL PROCESO DE MODELADO DE TÓPICOS
#Crea embeddings de las reviews
#embeddings = np.load(global_path)
embeddings = np.load(global_path)[valid_idx]

reducer = umap.UMAP(
    n_neighbors=30,
    n_components=5,
    metric="cosine"
)
print(f"\nReduce dimensiones\n")
embeddings_reduced = reducer.fit_transform(embeddings)


print(f"\nLimpia el dataset con isolation forest\n")
#Hace limpieza de outliers
filtered_emb, filtered_reviews, removed_idx, scores, stats = \
    remove_outliers_isolation_forest(embeddings_reduced, cleaned_reviews, contamination=0.05)


#Obtiene número adecuado de vecinos
best_guess = [int(np.log(len(filtered_emb)) * 3)]
#best_k, all_results = estimate_optimal_neighbors(filtered_emb, k_clusters=15, neighbor_range=range(best_guess[0]-3, best_guess[0]+3))
best_k = best_guess
print("Mejor número de vecinos:", best_k[0])

print(f"\nComienza clusterización con K-NN \n")
#Obtiene el grafo en base a las similitudes coseno
graph = build_similarity_graph(filtered_emb, int(best_k[0]))

#print(f"\nBusca resolución óptima\n")
#Obtiene resolución del grafo
labels, optimal_res = find_optimal_resolution(graph, filtered_emb, res_min=0.1, res_max=1.5, step=0.1, alpha=0.6)

#optimal_res = 0.1

if optimal_res is None:
    raise ValueError("Proceso detenido: No se pudo encontrar una resolución óptima para Leiden. Revise los parámetros de 'find_optimal_resolution'.")


print(f"\nCrea cluster mediante gráfos \n")
#Obtiene un clustering inicial
clusters = leiden_clustering(graph, resolution=optimal_res)

print(f"\nElimina clusteres muy pequeños \n")
#Elimina clusteres demasiado pequeños para ser considerados
(emb_final, texts_final, clusters_final, removed_clusters, cluster_stats) = filter_small_clusters(filtered_emb, filtered_reviews, clusters, min_cluster_size=int(len(filtered_emb) * 0.05))
print(f'\nCluster stats: {cluster_stats}\n')
print(f"\nMezcla centroides semánticamente\n")
#Mezcla los centroides en base a su similitud semántica y a la similitud de la matriz de similitud inter-cluster (solo el 5%)
labels_current = merge_clusters_hybrid(emb_final, clusters_final, percentile=85, alpha=0.6)

print(f"\nResultado del análisis \n")
#Obtiene las keywords para cada cluster
cluster_keywords = extract_cluster_keywordsKF(texts_final, emb_final, labels_current, top_docs_ratio=0.35, top_n=10)
pc,formated_llm_input = procesar_diccionario_topicos(cluster_keywords)


#pc = obtener_lista_palabras(cluster_keywords)
print(f'\nGuardando las KeyWords\n')
df = pd.DataFrame({'key':pc})
df.to_csv(keypat1)

#representative_docs = get_representative_documents_with_scores(emb_final, labels_current, texts_final, top_k=10)
#formatted_llm_input = format_llm_input_string(cluster_keywords, representative_docs)
print(f'\nPALABRAS CLAVE ENCONTRADAS\n')
print(f'\n{formated_llm_input}\n')

"""#PIPELINE"""

columna = 'opinion'
colecti = 'trip'
dataset = 'todo'
keypat1 = f'/content/drive/MyDrive/papers/verano2026/results/{colecti}/key-{dataset}.csv'
global_path = f'/content/drive/MyDrive/papers/verano2026/data/{colecti}/{dataset}.csv'
global_path2 = f'/content/drive/MyDrive/papers/verano2026/dataAux/{colecti}/'
model_path = f'/content/drive/MyDrive/papers/verano2026/LLM/mistral-7b-instruct-v0.2.Q4_K_M.gguf'

#CARGA DATASET
data = pd.read_csv(global_path, encoding='utf-8')


print(f"\nComienza proceso en dataset: {dataset} \n")
#Limpia Reviews
reviews = data[columna].tolist()
reviews_c = clean_reviews(reviews)

#Obtiene chunks
cleaned_reviews = recortaTodo(reviews_c)

# COMIENZA EL PROCESO DE MODELADO DE TÓPICOS
#Crea embeddings de las reviews
embeddings = embed_texts(cleaned_reviews)


reducer = umap.UMAP(
    n_neighbors=30,
    n_components=5,
    metric="cosine"
)
print(f"\nReduce dimensiones\n")
embeddings_reduced = reducer.fit_transform(embeddings)


print(f"\nLimpia el dataset con isolation forest\n")
#Hace limpieza de outliers
filtered_emb, filtered_reviews, removed_idx, scores, stats = \
    remove_outliers_isolation_forest(embeddings_reduced, cleaned_reviews, contamination=0.05)


#Obtiene número adecuado de vecinos
#best_guess = [int(np.log(len(filtered_emb)) * 3)]
#best_k, all_results = estimate_optimal_neighbors(filtered_emb, k_clusters=15, neighbor_range=range(best_guess[0]-5, best_guess[0]+5))
best_k = [27]
print("Mejor número de vecinos:", best_k[0])

print(f"\nComienza clusterización con K-NN \n")
#Obtiene el grafo en base a las similitudes coseno
graph = build_similarity_graph(filtered_emb, int(best_k[0]))

print(f"\nBusca resolución óptima\n")
#Obtiene resolución del grafo
labels, optimal_res = find_optimal_resolution(graph, filtered_emb, res_min=0.1, res_max=1.5, step=0.05, alpha=0.4)

if optimal_res is None:
    raise ValueError("Proceso detenido: No se pudo encontrar una resolución óptima para Leiden. Revise los parámetros de 'find_optimal_resolution'.")


print(f"\nCrea cluster mediante gráfos \n")
#Obtiene un clustering inicial
clusters = leiden_clustering(graph, resolution=optimal_res)

print(f"\nElimina clusteres muy pequeños \n")
#Elimina clusteres demasiado pequeños para ser considerados
(emb_final, texts_final, clusters_final, removed_clusters, cluster_stats) = filter_small_clusters(filtered_emb, filtered_reviews, clusters, min_cluster_size=int(len(filtered_emb) * 0.01))
print(f'\nCluster stats: {cluster_stats}\n')
print(f"\nMezcla centroides semánticamente\n")
#Mezcla los centroides en base a su similitud semántica y a la similitud de la matriz de similitud inter-cluster (solo el 5%)
labels_current = merge_clusters_hybrid(emb_final, clusters_final, percentile=95, alpha=0.6)

print(f"\nResultado del análisis \n")
#Obtiene las keywords para cada cluster
cluster_keywords = extract_cluster_keywords(texts_final, emb_final, clusters_final, top_docs_ratio=0.35, top_n=10)
pc = obtener_lista_palabras(cluster_keywords)
print(f'\nGuardando las KeyWords\n')
df = pd.DataFrame({'key':pc})
df.to_csv(keypat1)
print(f'\n{pc}\n')

#Obtiene evaluación de cohesión/separación de clusters
cluster_score = composite_cluster_score(emb_final, labels_current)
print(f'\nScores del proceso de\n')
print(f'Coherencia intra cluster: {cluster_score['C_intra']}\n')
print(f'Separación inter cluster: {cluster_score['S_inter']}\n')
print(f'Penalización: {cluster_score['Penalty']}\n')
print(f'Composite score: {cluster_score['Composite Score']}\n')

#Obtiene los documentos más representativos por cluster
representative_docs = get_representative_documents_with_scores(emb_final, labels_current, texts_final, top_k=2)

"""
#====================== LLM ==================================================

llm = Llama(
    model_path=model_path,
    n_ctx=4096,          # Tamaño del contexto
    n_gpu_layers=-1,     # Capas en GPU (-1 = todas si hay GPU)
    n_batch=512,         # Batch interno de tokens
    use_mlock=True,      # Evita swapping si hay RAM suficiente
    verbose=False
)

# Generate the formatted string for the LLM prompt
formatted_llm_input = format_llm_input_string(cluster_keywords, representative_docs)
strin = prompt + formatted_llm_input # Prepend the global prompt

print("\n\nPROMPT RESULTANTE:\n\n"+strin+"\n\n")
# If the LLM part is commented out, 'strin' might not be defined. Initialize it if it's going to be printed.
output = llm(strin, max_tokens=2048) # Increased max_tokens for a complete response
print(output['choices'][0]['text'])
"""
print(f"\n{formatted_llm_input}\n")

#Guarda variables para el análisis
np.save(global_path2+f"{dataset}-emb.npy", filtered_emb)
np.save(global_path2+f"{dataset}-rev.npy", filtered_reviews)
np.save(global_path2+f"{dataset}-clus.npy", labels_current)
print('\nFinalizado....\n')

print(output['choices'][0]['text'])

"""#CREA TODOS LOS EMBEDDINGS"""

#Configura dataset
columna = 'comment'
colecti = 'trip'
dataset = 'huatulco-T'
global_path = f'/content/drive/MyDrive/papers/verano2026/data/{colecti}/{dataset}.csv'
global_path2 = f'/content/drive/MyDrive/papers/verano2026/data/{colecti}/{dataset}.npy'
data = pd.read_csv(global_path, encoding='utf-8')


print(f"\nComienza proceso en dataset: {dataset} \n")
#Limpia Reviews
reviews = data[columna].tolist()
reviews_c = clean_reviews(reviews)

#Obtiene chunks
cleaned_reviews = recortaTodo(reviews_c)

# COMIENZA EL PROCESO DE MODELADO DE TÓPICOS
#Crea embeddings de las reviews
embeddings = embed_texts(cleaned_reviews)
np.save(global_path2, embeddings)
print('\nembeddings Guardados...\n')

