import torch
from torch.nn import Module, Parameter, init
import numpy as np
from einops import einsum, rearrange, repeat


class Linear(Module):

    def __init__(self, in_features: int, out_features: int, device: torch.device = None,
                 dtype: torch.dtype = None):
        """Construct a linear transformation module.

        Parameters
        ----------
        in_features : int
            final dimension of the input.
        out_features : int
            final dimension of the output.
        device : torch.device, optional
            Device to store the parameters on, by default None.
        dtype : torch.dtype, optional
            Data type of the parameters, by default None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = {"device": device, "dtype": dtype}
        self._init_params()

    def _init_params(self):
        w = torch.empty(size=(self.out_features, self.in_features), **self.kwargs)
        init.trunc_normal_(w, mean=0.0, std=np.sqrt(2 / (self.in_features + self.out_features)))
        self.W = Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input."""
        return einsum(self.W, x, "n m, ... m -> ... n")


class Embedding(Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None,
                 dtype: torch.dtype = None):
        """Construct an embedding module.

        Parameters
        ----------
        num_embeddings : int
            Size of the vocabulary
        embedding_dim : int
            Dimension of the embedding vectors, i.e. d_model
        device : torch.device, optional
            Device, by default None
        dtype : torch.dtype, optional
            Data type, by default None
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.kwargs = {"device": device, "dtype": dtype}
        self._init_params()

    def _init_params(self):
        w = torch.empty(size=(self.num_embeddings, self.embedding_dim), **self.kwargs)
        init.trunc_normal_(w, mean=0.0,
                           std=np.sqrt(2 / (self.num_embeddings + self.embedding_dim)))
        self.W = Parameter(w)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs."""
        return self.W[token_ids, :]


class RMSNorm(Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """Construct the RMSNorm module.

        This function should accept the following parameters:

        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.kwargs = {"device": device, "dtype": dtype}
        self._init_params()

    def _init_params(self):
        w = torch.empty(size=(self.d_model,), **self.kwargs)
        init.trunc_normal_(w, mean=0.0, std=np.sqrt(1 / self.d_model))
        self.W = Parameter(w)

    def _rms(self, x: torch.Tensor):
        a_sq = einsum(x, x, "... d_model, ... d_model -> ...")
        return torch.sqrt(1 / self.d_model * a_sq + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = self._rms(x)
        result = einsum(self.W, x, "d_model, ... d_model -> ... d_model")
        result = einsum(1 / rms, result, "..., ... d_model -> ... d_model")
        return result.to(in_dtype)


class SiLU(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(Module):

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """SwiGLU activation

        FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x)

        Parameters
        ----------
        d_model : int
            _description_
        d_ff : int
            _description_
        device : _type_, optional
            _description_, by default None
        dtype : _type_, optional
            _description_, by default None
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.kwargs = {"device": device, "dtype": dtype}
        self.silu = SiLU()
        self._init_params()

    def _init_params(self):
        w1 = torch.empty(size=(self.d_ff, self.d_model), **self.kwargs)
        w2 = torch.empty(size=(self.d_model, self.d_ff), **self.kwargs)
        w3 = torch.empty(size=(self.d_ff, self.d_model), **self.kwargs)

        self.W1 = Parameter(
            init.trunc_normal_(w1, mean=0.0, std=np.sqrt(2 / (self.d_ff + self.d_model)))
        )
        self.W3 = Parameter(
            init.trunc_normal_(w3, mean=0.0, std=np.sqrt(2 / (self.d_ff + self.d_model)))
        )
        self.W2 = Parameter(
            init.trunc_normal_(w2, mean=0.0, std=np.sqrt(2 / (self.d_ff + self.d_model)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU
        out = self.silu.forward(einsum(self.W1, x, "d_ff d_model, ... d_model -> ... d_ff"))
        # GLU
        out = out * einsum(self.W3, x, "d_ff d_model, ... d_model -> ... d_ff")
        return einsum(self.W2, out, "d_model d_ff, ... d_ff -> ... d_model")


class RotaryPositionalEmbedding(Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """Construct the RoPE module and create buffers if needed.

        Parameters
        ----------
        theta: float
            Θ value for the RoPE
        d_k: int
            dimension of query and key vectors
        max_seq_len: int
            Maximum sequence length that will be inputted
        device: torch.device | None = None
            Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        if d_k % 2 != 0:
            raise ValueError("The dimension has to be divisible by 2")
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.kwargs = {"device": device}
        self._init_params()

    def _init_params(self):

        def angle(i, k):
            return i * self.theta ** (2 * (1 - k) / self.d_k)

        r_buff = torch.zeros((self.max_seq_len, self.d_k, self.d_k), **self.kwargs)
        for i in range(self.max_seq_len):

            for k in range(1, self.d_k // 2 + 1):
                theta = angle(i, k)
                R_ik = torch.tensor([[np.cos(theta), -np.sin(theta)],
                                     [np.sin(theta), np.cos(theta)]])
                ind = 2 * (k - 1)
                r_buff[i, ind: ind + 2, ind: ind + 2] = R_ik

        self.register_buffer("rope_buffer", r_buff, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same
        shape.

        Note that you should tolerate x with an arbitrary number of batch dimensions. You should
        assume that the token positions are a tensor of shape (..., seq_len) specifying the token
        positions of x along the sequence dimension.
        You should use the token positions to slice your (possibly precomputed) cos and sin tensors
        along the sequence dimension.

        'tests/_snapshots/test_rope.npz'
        """
        # Repeat token positions over batches if necessary
        if len(x.shape) > len(token_positions.shape) + 1:
            batch_sizes = x.shape[:-2]
            batch_tp = token_positions
            for bs in batch_sizes:
                batch_tp = repeat(batch_tp, '... seq_len -> ... c seq_len', c=bs)

        R = self.rope_buffer[batch_tp, :, :]
        return einsum(R, x, "... seq_len i j, ... seq_len j -> ... seq_len i")
