from xidootopic.xidootopic import XidooTopic
from preprocessing.cleaner import TextCleaner
from preprocessing.chunks import TextChunker
from embedding.embedder import TextEmbedder
from embedding.encoder import HFencoder
from embedding.pooling import MeanPooling
from modeling.reduction import UMAPReducer
from modeling.outliers import IsolationForestOutlierRemover
from modeling.graph import KNNGraphBuilder
from modeling.clustering import LeidenResolutionOptimizer,LeidenClusterer
from modeling.postprocessing import ClusterFilter,ClusterMerger
from modeling.representation import KeyWordExtractor,RepresentativeDocsExtractor
from transformers import AutoModel, AutoTokenizer
import torch
from keybert import KeyBERT
import spacy
import nltk
import pandas as pd 
model_name = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model_hf = AutoModel.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_hf.to(device)
model_hf.eval()

df=pd.read_csv("../trip/huatulco.csv")
texts=df["comment"].tolist()
kw_model = KeyBERT(model=model_hf)  
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')

spanish_stopwords = nltk.corpus.stopwords.words('spanish')
model=XidooTopic(preprocessor=TextCleaner(),
                 chunker=TextChunker(),
                embedder=TextEmbedder(encoder=HFencoder(model=model_hf,
                                                        tokenizer=tokenizer,
                                                        device="cuda"),pooling=MeanPooling()),
                 reducer=UMAPReducer(),
                 outlier_detector=IsolationForestOutlierRemover(),
                 graph_builder=KNNGraphBuilder(),
                 optimal_resolution=LeidenResolutionOptimizer(),
                 clusterer=LeidenClusterer(),
                 cluster_filter=ClusterFilter(),
                 cluster_merger=ClusterMerger(),
                 keyword_model=KeyWordExtractor(kw_model=kw_model,
                                                stopwords=spanish_stopwords,
                                                nlp=nlp),
                doc_model= RepresentativeDocsExtractor())
result=model.fit_transform(texts)
print(result)
