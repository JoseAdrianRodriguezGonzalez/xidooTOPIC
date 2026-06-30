# tests/embedding/test_pooling.py

import torch
from embedding.pooling import MeanPooling

def test_mean_pooling_basic():
    pooling = MeanPooling()

    last_hidden = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]]
    ])  # (1, 2, 2)

    mask = torch.tensor([[1, 1]])
    result = pooling(last_hidden, mask)
    expected = torch.tensor([[2.0, 3.0]])

    assert torch.allclose(result, expected)


def test_mean_pooling_ignores_padding():
    pooling = MeanPooling()

    last_hidden = torch.tensor([
        [[1.0, 2.0], [100.0, 200.0]]
    ])

    mask = torch.tensor([[1, 0]])

    result = pooling(last_hidden, mask)

    expected = torch.tensor([[1.0, 2.0]])

    assert torch.allclose(result, expected)
