import torch
import math
from jaxtyping import Float, Int
from einops import einsum

class RMSNorm(torch.nn.Module):
    gain: Float[torch.Tensor, "d_model"]

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
        self.gain = torch.nn.Parameter(torch.ones(d_model))
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

        result = einsum(x, self.gain, "... d_model, d_model -> ... d_model")

        return result.to(in_dtype)
    
class SwiGLU(torch.nn.Module):
    w1_weight: Float[torch.Tensor, "d_ff d_model"]
    w2_weight: Float[torch.Tensor, "d_model d_ff"]
    w3_weight: Float[torch.Tensor, "d_ff d_model"]
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.w1_weight = torch.nn.Parameter(torch.empty(d_ff, d_model))
        self.w2_weight = torch.nn.Parameter(torch.empty(d_model, d_ff))
        self.w3_weight = torch.nn.Parameter(torch.empty(d_ff, d_model))
        self.d_model = d_model
        self.d_ff = d_ff

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        W1_x = einsum(self.w1_weight, x, "d_ff d_model, ... d_model -> ... d_ff")
        W3_x = einsum(self.w3_weight, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu_part = W1_x * torch.sigmoid(W1_x)
        dot_prod = silu_part * W3_x
        result = einsum(self.w2_weight, dot_prod, "d_model d_ff, ... d_ff -> ... d_model")
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