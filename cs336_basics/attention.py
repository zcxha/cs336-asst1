import math
import torch
import einx
from torch import Tensor
from jaxtyping import Float, Bool, Int
from einops import einsum
from cs336_basics.prenorm_transformer_block import RotaryPositionalEmbedding
def softmax(x: Float[Tensor, "..."], dim_i: int) -> Float[Tensor, "..."]:
    r"""
    apply the softmax operation on a tensor.
    Args:
        x (Tensor): input tensor
        dim_i (int): the dimension to apply softmax on
    """
    m = x.max(dim=dim_i, keepdim=True)
    x = x - m.values
    x = x.exp()
    L = x.sum(dim = dim_i, keepdim=True)
    return x / L

def scaled_dot_product_attention(
        Q: Float[Tensor, "batch_size ... seq_len d_k"],
        K: Float[Tensor, "batch_size ... k_len d_k"],
        V: Float[Tensor, "batch_size ... k_len d_v"],
        mask: Bool[Tensor, "... seq_len k_len"] | None = None
) -> Float[Tensor, "batch_size ... seq_len d_v"]:
    r"""
    attention
    """
    QK = einsum(Q, K, "... seq_len d_k, ... k_len d_k -> ... seq_len k_len") / math.sqrt(Q.shape[-1])

    if mask is not None:
        QK = QK.masked_fill(~mask, float("-inf"))

    softmaxed = softmax(QK, -1)

    return einsum(softmaxed, V, "... seq_len k_len, ... k_len d_v -> ... seq_len d_v") # (k_len = seq_len)  但是直接写会出现歧义

class CausalMultiHeadSelfAttention(torch.nn.Module):
    q_proj_weight: Float[Tensor, "(num_heads d_k) d_model"]
    k_proj_weight: Float[Tensor, "(num_heads d_k) d_model"]
    v_proj_weight: Float[Tensor, "(num_heads d_v) d_model"]
    o_proj_weight: Float[Tensor, "d_model (num_heads d_v)"]
    def __init__(self, d_model: int, num_heads: int, 
                 rope: Bool = False,
                 theta: float | None = None, 
                 max_seq_len: int | None = None, 
                ):
        super().__init__()
        self.q_proj_weight = torch.nn.Parameter(torch.empty(d_model, d_model))
        self.k_proj_weight = torch.nn.Parameter(torch.empty(d_model, d_model))
        self.v_proj_weight = torch.nn.Parameter(torch.empty(d_model, d_model))
        self.o_proj_weight = torch.nn.Parameter(torch.empty(d_model, d_model))
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope
        if rope:
            self.rpe = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
            

    def forward(self, x: Float[Tensor, "... sequence_length d_model"], token_positions: Int[Tensor, "... sequence_length"] | None = None) -> Float[Tensor, "... sequence_length d_model"]:
        seq_len = x.shape[-2]
        Q = einx.dot(
            "(num_heads d_k) d_model, ... sequence_length d_model -> ... num_heads sequence_length d_k", 
            self.q_proj_weight,
            x,
            num_heads = self.num_heads 
        )
        K = einx.dot(
            "(num_heads d_k) d_model, ... sequence_length d_model -> ... num_heads sequence_length d_k", 
            self.k_proj_weight,
            x,
            num_heads = self.num_heads
        )
        V = einx.dot(
            "(num_heads d_v) d_model, ... sequence_length d_model -> ... num_heads sequence_length d_v",
            self.v_proj_weight,
            x,
            num_heads = self.num_heads
        )

        if self.rope:
            assert(token_positions is not None)
            Q = self.rpe.forward(Q, token_positions)
            K = self.rpe.forward(K, token_positions)
        
        
        mask = ~torch.full((seq_len, seq_len), True, dtype=torch.bool).triu(1)

        mha = scaled_dot_product_attention(Q, K, V, mask)

        O = einx.dot(
            "d_model (num_heads d_v), ... num_heads sequence_length d_v -> ... sequence_length d_model",
            self.o_proj_weight,
            mha,
            num_heads = self.num_heads
        )

        return O