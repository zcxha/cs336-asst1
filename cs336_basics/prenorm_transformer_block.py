import torch
import math
import einx
from jaxtyping import Float, Int, Bool
from torch import Tensor
from einops import einsum
from cs336_basics.basic_blocks import Linear, Embedding
from cs336_basics.nn_utils import softmax
class RMSNorm(torch.nn.Module):
    weight: Float[torch.Tensor, "d_model"]

    def __init__(self, d_model: int, eps: float=1e-5, device: torch.device | None=None, dtype: torch.dtype | None=None):
        r"""
        Construct the RMSNorm module. This function should accept the following parameters:
        Args:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
        self.d_model = d_model


    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        r"""
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms_x = torch.sqrt(einsum(x, x, "... d_model, ... d_model -> ...") / self.d_model + self.eps)

        x = x / rms_x[:, :, None]

        result = einsum(x, self.weight, "... d_model, d_model -> ... d_model")

        return result.to(in_dtype)
    
class SwiGLU(torch.nn.Module):
    w1: Linear # Float[torch.Tensor, "d_ff d_model"]
    w2: Linear # Float[torch.Tensor, "d_model d_ff"]
    w3: Linear # Float[torch.Tensor, "d_ff d_model"]
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.d_model = d_model
        self.d_ff = d_ff

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        W1_x = self.w1(x)
        W3_x = self.w3(x)
        silu_part = W1_x * torch.sigmoid(W1_x)
        dot_prod = silu_part * W3_x
        result = self.w2(dot_prod)
        return result
        
class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        r"""
        Construct the RoPE module and create buffers if needed.
        Args:
            theta (float): Θ value for the RoPE
            d_k (int): dimension of query and key vectors
            max_seq_len (int): Maximum sequence length that will be input
            device (torch.device | None = None): Device to store the buffer on
        """
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        i = torch.arange(0, max_seq_len, 1, device=device).unsqueeze(1) # max_seq_len 1
        k = torch.arange(0, d_k // 2, 1, device=device).unsqueeze(0) # 1 d_k//2
        theta_i_k = i / (theta ** ((2 * k) / d_k))
        self.register_buffer("cos", torch.cos(theta_i_k), persistent=False)
        self.register_buffer("sin", torch.sin(theta_i_k), persistent=False)
        
    def forward(self, x: Float[torch.Tensor, "... sequence_length d_k"], 
                token_positions: Int[torch.Tensor, "... sequence_length"]
                ) -> Float[torch.Tensor, "... sequence_length d_k"]:
        cos = self.cos[token_positions] # ... sequence_length d_k // 2
        sin = self.sin[token_positions] # ... sequence_length d_k // 2
        even = x[..., ::2] # sequence_length d_k // 2
        odd = x[..., 1::2]
        rot_x = cos * even - sin * odd
        rot_y = sin * even + cos * odd 
        interleave_xy = torch.stack([rot_x, rot_y], dim=-1)
        result = interleave_xy.flatten(-2)
        return result

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
    # q_proj_weight: Float[Tensor, "(num_heads d_k) d_model"]
    # k_proj_weight: Float[Tensor, "(num_heads d_k) d_model"]
    # v_proj_weight: Float[Tensor, "(num_heads d_v) d_model"]
    # o_proj_weight: Float[Tensor, "d_model (num_heads d_v)"]
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    output_proj: Linear
    def __init__(self, d_model: int, num_heads: int, 
                 rope: Bool = False,
                 theta: float | None = None, 
                 max_seq_len: int | None = None, 
                 device: torch.device | None = None
                ):
        super().__init__()
        self.q_proj = Linear(d_model, d_model, device=device)
        self.k_proj = Linear(d_model, d_model, device=device)
        self.v_proj = Linear(d_model, d_model, device=device)
        self.output_proj = Linear(d_model, d_model, device=device)
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope
        self.device = device
        if rope:
            self.rpe = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device)
            

    def forward(self, x: Float[Tensor, "... sequence_length d_model"], token_positions: Int[Tensor, "... sequence_length"] | None = None) -> Float[Tensor, "... sequence_length d_model"]:
        seq_len = x.shape[-2]
        Q = einx.dot(
            "(num_heads d_k) d_model, ... sequence_length d_model -> ... num_heads sequence_length d_k", 
            self.q_proj.weight,
            x,
            num_heads = self.num_heads 
        )
        K = einx.dot(
            "(num_heads d_k) d_model, ... sequence_length d_model -> ... num_heads sequence_length d_k", 
            self.k_proj.weight,
            x,
            num_heads = self.num_heads
        )
        V = einx.dot(
            "(num_heads d_v) d_model, ... sequence_length d_model -> ... num_heads sequence_length d_v",
            self.v_proj.weight,
            x,
            num_heads = self.num_heads
        )

        if self.rope:
            assert(token_positions is not None)
            Q = self.rpe(Q, token_positions)
            K = self.rpe(K, token_positions)
        
        
        mask = ~torch.full((seq_len, seq_len), True, dtype=torch.bool, device=self.device).triu(1)

        mha = scaled_dot_product_attention(Q, K, V, mask)

        O = einx.dot(
            "d_model (num_heads d_v), ... num_heads sequence_length d_v -> ... sequence_length d_model",
            self.output_proj.weight,
            mha,
            num_heads = self.num_heads
        )

        return O

class TransformerBlock(torch.nn.Module):
    attn: CausalMultiHeadSelfAttention
    ln1: RMSNorm
    ffn: SwiGLU
    ln2: RMSNorm
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
        r"""
        Construct A TransformerBlock, initializing Modules
        Args:
            d_model (int): Dimensionality of the Transformer block inputs.
            num_heads (int): Number of heads to use in multi-head self-attention.
            d_ff (int): Dimensionality of the position-wise feed-forward inner layer.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
            theta (float): RoPE parameter.
        """
        super().__init__()
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, rope=True, theta=theta, max_seq_len=max_seq_len, device=device)
        self.ln1 = RMSNorm(d_model, device=device)
        self.ln2 = RMSNorm(d_model, device=device)
        self.ffn = SwiGLU(d_model, d_ff, device=device)
    
    def forward(self, x: Float[Tensor, "batch sequence_length d_model"]):
        seq_len = x.shape[1]
        batch_size = x.shape[0]
        token_positions = torch.arange(0, seq_len).unsqueeze(0).unsqueeze(0).expand([batch_size, 1, seq_len])
        
        attention_with_rope = self.attn(self.ln1(x), token_positions) + x

        ffn_result = self.ffn(self.ln2(attention_with_rope)) + attention_with_rope

        return ffn_result

class TransformerLM(torch.nn.Module):
    r"""
        LM
        Args:
            vocab_size(int): The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix.
            context_length(int): The maximum context length, necessary for determining the dimensionality of the RoPE sin and cos buffer.
            num_layers(int): The number of Transformer blocks to use.
            d_model (int): Dimensionality of the Transformer block inputs.
            num_heads (int): Number of heads to use in multi-head self-attention.
            d_ff (int): Dimensionality of the position-wise feed-forward inner layer.
            theta (float): RoPE parameter.
    """
    token_embeddings: Embedding
    layers: torch.nn.ModuleList
    ln_final: RMSNorm
    lm_head: Linear
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, theta: float, device: torch.device = torch.device("cuda"), dtype: torch.dtype | None = None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, context_length, theta, device=device, dtype=dtype) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, x: Int[Tensor, "batch_size sequence_length"]) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
        hidden: Float[Tensor, "batch_size sequence_length d_model"] = self.token_embeddings(x)

        for layer in self.layers:
            hidden = layer(hidden)
        
        hidden: Float[Tensor, "batch_size sequence_length d_model"] = self.ln_final(hidden)
        logits = self.lm_head(hidden)

        return logits