import collections
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
def get_topk_frequent(input_list,nlp):
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
class KeyWordExtractor:
    def __init__(self,kw_model,stopwords,nlp,top_docs_ratio:float=0.2,top_n:int=10):
        self.nlp=nlp 
        self.kw_model=kw_model
        self.stopwords=stopwords
        self.top_docs_ratio=top_docs_ratio
        self.top_n=top_n
    def extract(self,texts:list[str],embeddings:np.ndarray,labels:np.ndarray):
        cluster_keywords = {}
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_texts = np.array(texts)[cluster_mask]
            cluster_embs = embeddings[cluster_mask]

             # 1️⃣ Centroide
            centroid = cluster_embs.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid)+1e-9)
            sims = cosine_similarity(cluster_embs, centroid.reshape(1, -1)).flatten()
            n_top = max(1, int(len(sims) * self.top_docs_ratio))
            top_indices = sims.argsort()[-n_top:]

            listadocs =  cluster_texts[top_indices]
            topk = []
            resumen = " "
            for l in listadocs:
                kws=self.kw_model.extract_keywords(
                    l,
                    keyphrase_ngram_range=(1,1),
                    top_n=self.top_n,
                    vectorizer=CountVectorizer(stop_words=self.stopwords),
                    use_mmr=True,
                    diversity=0.7
                )
                topk.extend([k[0] for k in kws])
            if not topk:
                cluster_keywords[cluster_id] = ([], "[EMPTY]")
                continue
            keys = get_topk_frequent(topk,self.nlp)
            if len(keys) > 50:
                resumenFinal = (", ").join(keys[:50])
            else:
                resumenFinal = (",").join(keys)
            resumenFinal = "[KEYWORDS] "+resumenFinal
    
            resumen = resumenFinal #tok.decode(out[0], skip_special_tokens=True)
            cluster_keywords[cluster_id] = (keys[:10],resumen)
        return cluster_keywords
        
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
import torch 
class TopicSummarizer:
    def __init__(self,model_name,device="cpu"):
        self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        self.model=AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device) 
        if device=="cuda":
            self.model=self.model.half() 
    def summarize(self,text,max_length=128):
        inputs=self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
       
        
        with torch.inference_mode():
              out = self.model.generate(
                  **inputs,
                max_length=max_length,
                  num_beams=4,
                  repetition_penalty=1.2,
                  no_repeat_ngram_size=3,
                  early_stopping=True
              )
        return self.tokenizer.decode(out[0],skip_special_tokens=True)

       
