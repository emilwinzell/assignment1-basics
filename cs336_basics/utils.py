import os
import typing
import torch
import numpy as np
from einops import einsum
import functools


def _no_grad(func):
    """
    This wrapper is needed to avoid a circular import when using @torch.no_grad on the exposed
    functions
    """

    def _no_grad_wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    functools.update_wrapper(_no_grad_wrapper, func)
    return _no_grad_wrapper


def softmax(x: torch.Tensor, i: int, tau: float = 1.0) -> torch.Tensor:
    """Apply softmax to tensor on a specified dimension

    Parameters
    ----------
    x : torch.Tensor
    i : int
        The dimension.
    tau : float, optional
        Temperature scaling.

    Returns
    -------
    torch.Tensor
    """
    tau = max(tau, 0.01)  # Minimum value for numerical stability
    out = (x - x.max(dim=i, keepdim=True)[0]) / tau
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


def cross_entropy_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    predicted logits (oi) and targets (xi+1) and computes the cross
    entropy ℓi = − log softmax(oi)[xi+1]

    preds = (batch_size ..., vocab_size), targets = (batch_size ...)
    -ln (softmax(oi)) = ln(sum(exp(o))) - oi
    """
    out = preds - preds.max(dim=-1, keepdim=True)[0]
    # One hot encode targets to retrieve logits with einsum
    one_hot = torch.eye(out.shape[-1], device=preds.device)[targets]
    batch_loss = torch.log(
        torch.sum(out.exp(), dim=-1)
    ) - einsum(out, one_hot, "... vs, ... vs -> ...")
    if batch_loss.dim() == 1:
        batch_loss = torch.unsqueeze(batch_loss, 0)  # Add a dimension to take mean over
    return batch_loss.mean()  # Mean over all except first batch dimension


def cosine_annealing_lr_schedule(t: int, lr_max: float, lr_min: float, warm_up: int,
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


@_no_grad
def gradient_clipping(params: list[torch.nn.Parameter], max_norm: float, eps: float = 1e-6):
    all_norms = []
    for param in params:
        if param.grad is None:
            continue
        grad = param.grad
        grad_l2 = torch.linalg.norm(grad, ord=2)
        all_norms.append(grad_l2)

    total_norm = torch.linalg.norm(torch.stack(all_norms), ord=2)
    if total_norm < max_norm:
        return

    coef = max_norm / (total_norm + eps)
    for param in params:
        if param.grad is None:
            continue
        param.grad.mul_(coef)

    return total_norm.item()


def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str = None,
              pad_val: int = 256) -> tuple[torch.Tensor, torch.Tensor]:

    size = x.shape[0]
    batch_end_ind = context_length + batch_size + 1
    if size < batch_end_ind:
        x_padded = np.pad(x, ((0, batch_end_ind - size),), mode='constant',
                          constant_values=pad_val)
        size = x_padded.shape[0]
    else:
        x_padded = x

    batch = torch.empty(size=(batch_size, context_length), device=device, dtype=int)
    targets = torch.empty(size=(batch_size, context_length), device=device, dtype=int)

    start_ind = 0
    for i in range(batch_size):
        if size > batch_end_ind:
            start_ind = np.random.randint(low=0, high=size - context_length)

        batch[i, :] = torch.from_numpy(x_padded[start_ind: start_ind + context_length])
        targets[i, :] = torch.from_numpy(x_padded[start_ind + 1: start_ind + context_length + 1])

    return batch, targets


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int,
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    """Dumps all the states into the file-like object out."""
    states = {"model": model.state_dict(),
              "optimizer": optimizer.state_dict(),
              "iteration": iteration}
    torch.save(states, out)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """Load checkpoint form src."""
    try:
        states = torch.load(src)
    except RuntimeError:
        states = torch.load(src, map_location=torch.device('cpu'), weights_only=False)

    model.load_state_dict(states["model"])
    optimizer.load_state_dict(states["optimizer"])
    return states["iteration"]
