import torch
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


def test_encoder_output_shape():
    model = DummyModel()
    tokenizer = DummyTokenizer()
    encoder = HFencoder(model, tokenizer, device="cpu")
    hidden, mask = encoder.encode(["hola", "mundo"])

    assert hidden.shape == (2, 5, 4)
    assert mask.shape == (2, 5)
