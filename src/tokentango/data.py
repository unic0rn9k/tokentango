import torch
from dataclasses import dataclass

# For now, we test using source_tokens as input,
# meaning we technically don't use masked_tokens in model validation
@dataclass
class BertData:
    source_tokens: torch.Tensor
    masked_tokens: torch.Tensor
    labels: torch.Tensor
