import numpy as np
import torch

from embedding.embedder import TextEmbedder
from embedding.pooling import MeanPooling
from embedding.encoder import HFencoder

class DummyModel:
    def __call__(self, **kwargs):
        batch = kwargs["input_ids"].shape[0]
        seq = kwargs["input_ids"].shape[1]

        return type("Output", (), {
            "last_hidden_state": torch.ones((batch, seq, 4))
        })
class DummyTokenizer:
    def __call__(self, texts, **kwargs):
        batch_size = len(texts)
        seq_len = 5
        return {
            "input_ids": torch.ones((batch_size, seq_len), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long)
        }


def test_embedder_output_shape():
    encoder = HFencoder(DummyModel(), DummyTokenizer(), device="cpu")
    pooling = MeanPooling()
    embedder = TextEmbedder(encoder, pooling, batch_size=2)

    texts = ["hola mundo", "otro texto", "más texto"]
    emb = embedder.embed_texts(texts)

    assert isinstance(emb, np.ndarray)
    assert emb.shape[0] == 3


def test_embedder_normalization():
    encoder = HFencoder(DummyModel(), DummyTokenizer(), device="cpu")
    pooling = MeanPooling()

    embedder = TextEmbedder(encoder, pooling, normalize=True)

    emb = embedder.embed_texts(["hola"])

    norm = np.linalg.norm(emb, axis=1)

    assert np.allclose(norm, 1.0)
