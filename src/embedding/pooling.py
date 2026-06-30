from torch import clamp
import torch

class MeanPooling:
    def __call__(self, last_hidden_state:torch.Tensor, attention_mask:torch.Tensor):
        #(batch, seq) -> (batch, seq, hidden_dim)
        mask=attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        #(batch_size, hidden_dim)
        summed=(last_hidden_state*mask).sum(1)
        counts=mask.sum(1).clamp(min=1e-9)
        return summed / counts


