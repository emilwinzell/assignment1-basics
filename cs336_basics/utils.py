import torch
import numpy as np
from einops import einsum


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    """Apply softmax to tensor on a specified dimension

    Parameters
    ----------
    x : torch.Tensor
    i : int
        The dimension.

    Returns
    -------
    torch.Tensor
    """
    out = x - x.max(dim=i, keepdim=True)[0]
    return out.exp() / torch.sum(out.exp(), dim=i, keepdim=True)


def scaled_dot_product_attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                                 mask: torch.Tensor = None) -> torch.Tensor:
    """Scaled dot product attention.

    Parameters
    ----------
    queries : torch.Tensor
        Shape (batch_size, ..., seq_len, d_k)
    keys : torch.Tensor
        Shape (batch_size, ..., seq_len, d_k)
    values : torch.Tensor
        Shape (batch_size, ..., seq_len, d_v)
    mask : torch.Tensor, optional
        Attention mask with shape (seq_len, seq_len), by default None

    Returns
    -------
    torch.Tensor
        Attention, shape (batch_size, ..., d_v)
    """
    d_k = queries.shape[-1]
    out = einsum(queries, keys, "b ... i d_k, b ... j d_k -> b ... i j") / np.sqrt(d_k)

    if mask is not None:
        out[..., ~mask] = -torch.inf

    out = softmax(out, -1)
    return einsum(out, values, "b ... i s, b ... s d_v -> b ... i d_v")


def cross_entropy_loss(preds: torch.Tensor, targets: torch.Tensor):
    """
    predicted logits (oi) and targets (xi+1) and computes the cross entropy ℓi = − log softmax(oi)[xi+1]

    preds = (batch_size ..., vocab_size), targets = (batch_size ...)
    -ln (softmax(oi)) = ln(sum(exp(o))) - oi
    """
    out = preds - preds.max(dim=-1, keepdim=True)[0]
    one_hot = torch.eye(out.shape[-1])[targets]  # One hot encode targets to retrieve logits with einsum
    batch_loss = torch.log(torch.sum(out.exp(), dim=-1)) - einsum(out, one_hot, "... vs, ... vs -> ...")
    if batch_loss.dim() == 1:
        batch_loss = torch.unsqueeze(batch_loss, 0)  # Add a dimension to take mean over
    return batch_loss.flatten(1).mean(1)  # Mean over all except first batch dimension


def learning_rate_schedule(t: int, lr_max: float, lr_min: float, warm_up: int,
                           annealing: int) -> float:
    """Cosine annealing learning rate schedule

    Parameters
    ----------
    t : int
        Iteration
    lr_max : float
        Learning rate max
    lr_min : float
        Learning rate min
    warm_up : int
        Number of warm up iterations, at which t should warm-up stop.
    annealing : int
        Number of annealing iterations, at which t should annealing stop.

    Returns
    -------
    float
        Learning rate at step t
    """
    if annealing <= warm_up:
        raise ValueError("The number of annealing iterations must be greater than warm_up")
    if t < warm_up:
        # Warm up phase
        return t / warm_up * lr_max
    elif t > annealing:
        # Post annealing phase
        return lr_min
    else:
        w = np.pi * (t - warm_up) / (annealing - warm_up)
        return lr_min + 0.5 * (1 + np.cos(w)) * (lr_max - lr_min)


def gradient_clipping(params: list[torch.nn.Parameter], max_norm: float, eps: float = 1e-6):
    all_norms = []
    for param in params:
        if param.grad is None:
            continue
        grad = param.grad.data
        grad_l2 = torch.linalg.norm(grad, ord=2)
        all_norms.append(grad_l2)

    total_norm = torch.linalg.norm(torch.stack(all_norms), ord=2)
    if total_norm < max_norm:
        return

    for param in params:
        if param.grad is None:
            continue
        param.grad.data = grad.mul(max_norm / (total_norm + eps))
