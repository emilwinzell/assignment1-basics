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
                                 mask: torch.Tensor = None):
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
