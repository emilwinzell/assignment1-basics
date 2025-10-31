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
        weight = torch.empty(size=(self.out_features, self.in_features), **self.kwargs)
        std = np.sqrt(2 / (self.in_features + self.out_features))
        init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        self.weight = Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input."""
        return einsum(self.weight, x, "n m, ... m -> ... n")


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
        weight = torch.empty(size=(self.num_embeddings, self.embedding_dim), **self.kwargs)
        init.trunc_normal_(weight, mean=0.0, std=1, a=-3, b=3)
        self.weight = Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs."""
        return self.weight[token_ids, :]


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
        weight = torch.ones(size=(self.d_model,), **self.kwargs)
        self.weight = Parameter(weight)

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
        result = einsum(self.weight, x, "d_model, ... d_model -> ... d_model")
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
        self.w1 = Linear(in_features=self.d_model, out_features=self.d_ff, **self.kwargs)
        self.w2 = Linear(in_features=self.d_ff, out_features=self.d_model, **self.kwargs)
        self.w3 = Linear(in_features=self.d_model, out_features=self.d_ff, **self.kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU
        out = self.silu.forward(self.w1.forward(x))
        # GLU
        out = out * self.w3.forward(x)
        return self.w2.forward(out)


class RotaryPositionalEmbedding(Module):
    theta: float
    d_k: int
    max_seq_len: int

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

    def set_max_seq_len(self, max_seq_len: int):
        if max_seq_len != self.max_seq_len:
            self.max_seq_len = max_seq_len
            self._init_params()

    def set_d_k(self, d_k: int):
        if d_k != self.d_k:
            if d_k % 2 != 0:
                raise ValueError("The dimension has to be divisible by 2")
            self.d_k = d_k
            self._init_params()

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
        batch_tp = torch.squeeze(token_positions)
        # Repeat token positions over batches if necessary
        if len(x.shape) > len(batch_tp.shape) + 1:
            batch_sizes = x.shape[:-2]
            for bs in batch_sizes:
                batch_tp = repeat(batch_tp, '... seq_len -> ... c seq_len', c=bs)

        R = self.rope_buffer[batch_tp, :, :]
        return einsum(R, x, "... seq_len i j, ... seq_len j -> ... seq_len i")
