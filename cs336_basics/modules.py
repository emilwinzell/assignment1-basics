import torch
from torch.nn import Module, Parameter, init
import numpy as np
from einops import einsum, rearrange

class Linear(Module):

    def __init__(self, in_features: int, out_features: int, device: torch.device = None, dtype: torch.dtype = None):
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

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None, dtype: torch.dtype = None):
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
        init.trunc_normal_(w, mean=0.0, std=np.sqrt(2 / (self.num_embeddings + self.embedding_dim)))
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

if __name__=='__main__':

    x = torch.tensor(np.ones(10))
    y = Linear(10, 5, dtype=float).forward(x)
    import pdb; pdb.set_trace()