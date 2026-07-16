import numpy as np 
import torch

from embedding.encoder import HFencoder
from embedding.pooling import MeanPooling 
class TextEmbedder:
    @classmethod
    def default(cls, model_name="sentence-transformers/all-MiniLM-L6-v2"):

        from transformers import AutoModel, AutoTokenizer
        device="cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_hf = AutoModel.from_pretrained(model_name)

        encoder = HFencoder(model_hf,tokenizer,device)
        pooling = MeanPooling()
        model_hf.to(device)
        model_hf.eval()
        
        return cls(
            encoder=encoder,
            pooling=pooling,
            batch_size=32,
            normalize=True,
            use_prefix=False
        )
    def __init__(self,encoder:HFencoder,
                 pooling:MeanPooling,
                 batch_size:int=32,
                 normalize:bool=True,
                 use_prefix:bool=False):
        self.encoder = encoder

        self.pooling = pooling
        self.batch_size = batch_size
        self.normalize = normalize
        self.use_prefix = use_prefix
    def _prepare_texts(self,texts:list[str])->list[str]:
        if self.use_prefix:
            return ["passage: " + t for t in texts]
        return texts 
    def embed_texts(self,texts):
        outs = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            batch_texts=self._prepare_texts(batch_texts)
            hidden,mask=self.encoder.encode(batch_texts)
            emb = self.pooling(hidden, mask)
            if self.normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)

            outs.append(emb.cpu().numpy())
        return np.vstack(outs)

    
