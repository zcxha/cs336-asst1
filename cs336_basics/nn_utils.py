import torch
import math
from torch import Tensor
from jaxtyping import Float, Int
from collections.abc import Callable
from typing import Iterable, Optional
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

def cross_entropy(logits: Float[Tensor, "... vocab_size"], targets: Int[Tensor, "..."]) -> Float[Tensor, ""]:
    r"""
    calculate cross entropy of logits and targets

    for one vector o_i and target x_{i+1}, l_i = - log softmax(o_i)[x_{i+1}] = - log (exp(o_i[x_{i+1}])/ sum(exp(o_i{a})) ) = log(sum()) - o_i{x_{i+1}}
    """
    m = logits.max(dim=-1, keepdim=True)
    logits = logits - m.values
    S = logits.exp().sum(dim=-1, keepdim=True) # ... 1
    L = S.log() - logits.gather(dim=-1, index=targets.unsqueeze(-1))
    return L.mean()

class AdamW(torch.optim.Optimizer):
    r"""
    Args:
        params: Model Params to Optimize
        alpha: learning rate

    """
    def __init__(self, params, lr, weight_decay,  betas: tuple, eps):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"alpha": lr, "beta1": betas[0], "beta2": betas[1], "epsilon": eps, "lamda": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["alpha"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            epsilon = group["epsilon"]
            lamda = group["lamda"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                grad = p.grad.data
                alpha_t = alpha * (math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))
                p.data -= alpha * lamda * p.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                p.data -= alpha_t * m / (torch.sqrt(v) + epsilon)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss

def get_lr_cosine_schedule(it: int, max_learning_rate: float, min_learning_rate: int, warmup_iters: int, cosine_cycle_iters: int) -> float:
    if it < warmup_iters:
        return float(it) / warmup_iters * max_learning_rate
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos(float(it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    l2_norm = 0
    for p in parameters:
        if p.grad is None:
            continue
        n = torch.linalg.norm(p.grad) 
        l2_norm += n * n
    l2_norm  = math.sqrt(l2_norm)

    scale = (l2_norm + 1e-6) / max_l2_norm

    if scale <= 1:
        return
    
    for p in parameters:
        if p.grad is None:
            continue
        p.grad.mul_(1 / scale)
    