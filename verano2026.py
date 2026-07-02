

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

from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModel, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM

"""#Funciones"""

#Obtiene los K documentos más representativos


#Obtiene todas las palabras clave del diccionario


#EVALUACIÓN DE LA COHESION INTRA Y SEPARACIÓN DE CLUSTER BASADO EN EMBEDDINGS Y SIM COSENO


# AUXILIAR DE LLM ===============================================================


#BM25 MODIFICADO PARA OBTENER KEYWORDS ================================================




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

