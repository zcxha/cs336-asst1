import torch
from jaxtyping import Float, Int
from torch import Tensor
from cs336_basics.prenorm_transformer_block import TransformerLM
def softmax_with_temperature(x: Float[Tensor, "..."], dim_i: int, temperature: float) -> Float[Tensor, "..."]:
    r"""
    apply the softmax operation on a tensor. with temperature
    Args:
        x (Tensor): input tensor
        dim_i (int): the dimension to apply softmax on
    """
    m = x.max(dim=dim_i, keepdim=True)
    x = x - m.values
    x = x / temperature
    x = x.exp()
    L = x.sum(dim = dim_i, keepdim=True)
    return x / L

def top_p_distribution(x: Float[Tensor, "... vocab_size"], top_p: float) -> Float[Tensor, "... vocab_size"]:
    sorted_probs, indices = torch.sort(x, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    shifted_cum_probs = torch.cat([
        torch.zeros_like(cumulative_probs[...,:1]),
        cumulative_probs[..., :-1]
    ], dim=-1)
    zero_mask = shifted_cum_probs >= top_p
    sorted_probs[zero_mask] = 0
    return x.scatter(-1, indices, sorted_probs)
    

def decoding(model: TransformerLM, prompt: list[int], temperature: float, top_p: float, max_tokens_gen: int, eos_token_id: int, device: torch.device = "cuda") -> list[int]:
    r"""
    Takes input prompts indices(maybe batched), return generated indices(maybe batched)

    Args:
        model: the TransformerLM to generate text
        prompt: input prompt
        temperature: temp
        top_p: top_p
        max_tokens_gen: maximum number of generated tokens
        eos_token_id: end of text token id.
    """
    response = []
    prompt = torch.tensor(prompt, device=device).unsqueeze(0) # 1 sequence_length
    
    for i in range(max_tokens_gen):
        logit = model(prompt)[:, -1, :]

        prob_distribution = softmax_with_temperature(logit, -1, temperature=temperature)

        prob_distribution = top_p_distribution(prob_distribution, top_p)

        next_token = prob_distribution.multinomial(1)

        if eos_token_id == next_token:
            break

        response.append(next_token.item())

        prompt = torch.cat([
            prompt,
            next_token,
        ], dim=-1)

    return response
