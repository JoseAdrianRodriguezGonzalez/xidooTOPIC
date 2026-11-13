# XidooTopic

**Graph-based topic modeling using embeddings, Leiden clustering, and semantic merging**

---

## 🚀 Overview

XidooTopic is a modular framework for unsupervised topic modeling built on top of modern NLP embeddings and graph-based clustering. It transforms raw text corpora into coherent, semantically meaningful topics through a pipeline that combines:

- Transformer-based embeddings  
- Dimensionality reduction (UMAP)  
- Outlier detection (Isolation Forest)  
- Graph construction (k-NN via FAISS)  
- Community detection (Leiden algorithm)  
- Semantic cluster merging  
- Keyword extraction (KeyBERT)  

---

## 📦 Installation

```bash
pip install xidootopic
```

Optional dependencies:

```bash
pip install xidootopic[nlp]
pip install xidootopic[llm]
```

---

## 🧠 Pipeline Summary

The method follows a structured six-stage pipeline:

1. Text Preprocessing  
   Raw reviews are cleaned, normalized, and filtered to remove noise and low-information samples.

2. Embedding Generation  
   Each document is mapped into a dense semantic vector space using transformer-based models.

3. Dimensionality Reduction  
   UMAP reduces embeddings to a low-dimensional manifold while preserving local structure.

4. Outlier Detection  
   Isolation Forest removes anomalous samples that may distort clustering structure.

5. Graph Construction & Clustering  
   A k-NN graph is built using FAISS and clustered using the Leiden algorithm with resolution optimization.

6. Postprocessing  
   Clusters are refined via:
   - Removal of small clusters  
   - Semantic merging of similar clusters  
   - Keyword extraction with KeyBERT  

---

## 📊 Core Ideas Behind the Method

### Isolation Forest

> [!NOTE]
> Isolation Forest assumes anomalies are easier to isolate than normal points. It builds random partition trees and assigns higher anomaly scores to points that require fewer splits to isolate.

This makes it highly effective for high-dimensional embeddings where distance-based methods degrade.

---

### Leiden Clustering

> [!NOTE]
> Leiden improves over Louvain by guaranteeing connected communities and better modularity optimization.
> It operates in three phases:
> - Local node movement  
> - Partition refinement  
> - Graph aggregation  
> This results in more stable and coherent topic clusters.

---

### Semantic Cluster Merging

After initial clustering, clusters are merged based on:

- Cosine similarity between centroids  
- Cluster cohesion  
- Adaptive thresholds combining global and local structure  

> [!NOTE]
> This step reduces redundancy and improves topic interpretability by merging semantically overlapping clusters.

---

### Keyword Extraction (KeyBERT)

> [!NOTE]
> KeyBERT selects keywords based on embedding similarity rather than frequency.

> Unlike TF-IDF, KeyBERT captures semantic relevance, allowing rare but meaningful terms to surface as keywords.

Implementation details:

- Only top 35% most representative documents per cluster are used  
- MMR is applied to ensure diversity  
- Results are lemmatized and aggregated  

---

## 🧪 Features

- Modular architecture for experimentation  
- Graph-based clustering pipeline  
- Automatic resolution tuning for Leiden  
- Semantic cluster merging  
- Robust outlier filtering  
- KeyBERT-based topic representation  

---
## Execution example
```python
from xidootopic.xidootopic import XidooTopic
#Can be any list of strings
texts = [
    "La inteligencia artificial está transformando la industria tecnológica.",
    "Python es uno de los lenguajes más utilizados para desarrollar aplicaciones de machine learning.",
    "Las redes neuronales profundas permiten identificar patrones complejos en grandes conjuntos de datos.",
    "El procesamiento de lenguaje natural facilita la interacción entre humanos y computadoras.",
    "Los modelos de lenguaje grandes pueden generar textos, responder preguntas y resumir documentos.",
    "La computación en la nube permite ejecutar aplicaciones sin administrar infraestructura física.",
    "Los sistemas distribuidos mejoran la escalabilidad y disponibilidad de los servicios.",
    "El aprendizaje automático se utiliza para detectar fraudes financieros.",
    "La Universidad Nacional Autónoma de México desarrolla investigaciones científicas.",
    "El cambio climático representa uno de los mayores desafíos ambientales actuales.",
    "La medicina moderna utiliza inteligencia artificial para apoyar el diagnóstico de enfermedades.",
    "Los investigadores estudian nuevas formas de producir energía renovable.",
    "La educación digital facilita el acceso al conocimiento desde diferentes regiones.",
    "México cuenta con una gran diversidad cultural, histórica y lingüística.",
    "La exploración espacial permite estudiar planetas y galaxias distantes.",
    "Los avances en biotecnología han mejorado diversos procesos agrícolas.",
    "Estoy muy feliz porque finalmente terminé mi proyecto de investigación.",
    "Me encanta aprender cosas nuevas y desarrollar soluciones innovadoras.",
    "Este producto es excelente y superó todas mis expectativas.",
    "Estoy decepcionado porque el servicio fue lento y poco eficiente.",
    "La experiencia fue terrible, definitivamente no volvería a comprar aquí.",
    "No estoy seguro de que esta propuesta sea realmente útil.",
    "El resultado fue aceptable, aunque todavía existen muchos aspectos por mejorar.",
    "Me siento frustrado porque el programa sigue presentando errores.",
    "La Universidad de Guanajuato se encuentra en el estado de Guanajuato, México.",
    "aunque necesitan muchos recursos computacionales.",
    "El análisis de datos permite descubrir patrones que no son evidentes."
]
model=XidooTopic.default()

result=model.fit_transform(texts)
print(result)
```


## 📁 Project Structure

```text
src/
├── embedding/
├── preprocessing/
├── modeling/
├── xidootopic/
```

---

## ⚙️ Configuration Example

```python
from xidootopic.xidootopic import xidootopic

model = xidootopic(
    preprocessor=TextCleaner(),
    chunker=TextChunker(),
    embedder=TextEmbedder(...),
    reducer=UMAPReducer(),
    outlier_detector=IsolationForestOutlierRemover(),
    graph_builder=KNNGraphBuilder(),
    clusterer=LeidenClusterer(),
    cluster_merger=ClusterMerger(),
    keyword_model=KeyWordExtractor(...)
)

result = model.fit_transform(texts)
```

---

## 📌 Key Contributions

- Graph-based topic modeling pipeline  
- Hybrid global/local cluster merging strategy  
- Efficient embedding-based keyword extraction  
- Fully modular design for research and experimentation  

---

## 📚 Methodological Foundations

- Isolation Forest — Liu et al. (2008, 2012)  
- Leiden Algorithm — Traag et al. (2019)  
- KeyBERT — Grootendorst (2020)  
- UMAP — McInnes et al. (2018)  

---

## 🧾 Summary of Process

- Clean text data  
- Generate embeddings  
- Reduce dimensionality  
- Remove outliers  
- Build similarity graph  
- Apply Leiden clustering  
- Optimize resolution  
- Filter small clusters  
- Merge semantically similar clusters  
- Extract keywords and representations  

---

## 📌 License

MIT License  

---

## ✨ Status

> [!NOTE]
> This project is currently in beta stage, designed for research and experimentation in topic modeling and semantic clustering.
```
