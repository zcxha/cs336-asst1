import torch
import math
from jaxtyping import Float
from einops import einsum
class Linear(torch.nn.Module):
    weight: Float[torch.Tensor, "d_out d_in"]
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        r"""
        Construct a linear transformation module. This function should accept the following parameters:
        Args:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, std=std, a = -3 * std, b = 3 * std)
    
    def forward(self, x: Float[torch.Tensor, "... d_in"]) -> Float[torch.Tensor, "... d_out"]:
        r"""
        Apply the linear transformation to the input.

        W shape "d_out d_in"
        Args:
            x (Float[torch.Tensor, "... d_in"]):
        Return:
            "... d_out" the W * x result
        """
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")

class Embedding(torch.nn.Module):
    weight: Float[torch.Tensor, "vocab_size d_model"]
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        r"""
        Construct an embedding module. This function should accept the following parameters:
        Args:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., 𝑑model
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        torch.nn.init.trunc_normal_(self.weight,mean=0, std=1, a = -3, b = 3)
    
    def forward(self, token_ids: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "... d_model"]:
        r"""
        Lookup the embedding vectors for the given token IDs.
        """
        return self.weight[token_ids]