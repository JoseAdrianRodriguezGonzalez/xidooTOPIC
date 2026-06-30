import torch
class HFencoder:
    def __init__(self,model,tokenizer,device,max_length=512):
        self.model=model
        self.tokenizer=tokenizer
        self.device=device
        self.max_length=max_length

    def encode(self,texts):
        enc=self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        if hasattr(enc,"to"):
            enc=enc.to(self.device)
        else:
            enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs=self.model(**enc)
        return outputs.last_hidden_state,enc["attention_mask"]



