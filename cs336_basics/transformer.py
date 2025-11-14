import torch
from torch.nn import Module
from einops import rearrange

from .utils import scaled_dot_product_attention
from .modules import RotaryPositionalEmbedding, Linear, RMSNorm, SwiGLU, Embedding


class MultiHeadSelfAttention(Module):

    def __init__(self, d_model: int, num_heads: int, device: torch.device = None,
                 dtype: torch.dtype = None, rope: RotaryPositionalEmbedding = None):
        """Construct a causal multi-head self-attention module.

        Parameters
        ----------
        d_model : int
        num_heads : int
        device : torch.device, optional
            Device to store the parameters on, by default None.
        dtype : torch.dtype, optional
            Data type of the parameters, by default None.
        """
        super().__init__()
        self._d_model = d_model
        self._dkv = d_model // num_heads
        self._h = num_heads
        if self._d_model != self._dkv * num_heads:
            raise ValueError("d_model must be divisible by number of heads")
        self.rope = rope

        if self.rope:
            self.rope.set_d_k(self._dkv)

        self.kwargs = {"device": device, "dtype": dtype}
        self._init_params()

    def _init_params(self):
        self.q_proj = Linear(in_features=self._d_model,
                             out_features=self._h * self._dkv,
                             **self.kwargs)
        self.k_proj = Linear(in_features=self._d_model,
                             out_features=self._h * self._dkv,
                             **self.kwargs)
        self.v_proj = Linear(in_features=self._d_model,
                             out_features=self._h * self._dkv,
                             **self.kwargs)
        self.output_proj = Linear(in_features=self._h * self._dkv,
                                  out_features=self._d_model,
                                  **self.kwargs)

    def forward(self, x: torch.Tensor, tk_pos: torch.Tensor = None) -> torch.Tensor:
        # TODO: use one weight matrix
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones((seq_len, seq_len))).to(dtype=bool)

        # Projections
        wqx = self.q_proj.forward(x)
        wkx = self.k_proj.forward(x)
        wvx = self.v_proj.forward(x)

        # Rearrange into (batch_size, num_heads, seq_len, d_k)
        wqx = rearrange(wqx, "... seq_len (h d_k) -> ... h seq_len d_k", h=self._h)
        wkx = rearrange(wkx, "... seq_len (h d_k) -> ... h seq_len d_k", h=self._h)
        wvx = rearrange(wvx, "... seq_len (h d_k) -> ... h seq_len d_k", h=self._h)

        if self.rope and tk_pos is not None:
            # self.rope.set_max_seq_len(seq_len)
            wqx = self.rope.forward(wqx, tk_pos)
            wkx = self.rope.forward(wkx, tk_pos)

        out = scaled_dot_product_attention(wqx, wkx, wvx, mask)
        # Merge heads again
        out = rearrange(out, "... h seq_len d_k -> ... seq_len (h d_k)")

        return self.output_proj(out)


class TransformerBlock(Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, device: torch.device = None,
                 dtype: torch.dtype = None, rope: RotaryPositionalEmbedding = None):
        super().__init__()
        self._d_model = d_model
        self._h = num_heads
        self._d_ff = d_ff
        self.kwargs = {"device": device, "dtype": dtype}
        self._init_params(rope)

    def _init_params(self, rope: RotaryPositionalEmbedding):
        self.attn = MultiHeadSelfAttention(d_model=self._d_model,
                                           num_heads=self._h,
                                           rope=rope,
                                           **self.kwargs)
        self.ln1 = RMSNorm(d_model=self._d_model, **self.kwargs)
        self.ln2 = RMSNorm(d_model=self._d_model, **self.kwargs)
        self.ffn = SwiGLU(d_model=self._d_model,
                          d_ff=self._d_ff,
                          **self.kwargs)

    def forward(self, x: torch.Tensor):
        (batch_size, seq_len, d_model) = x.shape
        if d_model != self._d_model:
            raise ValueError("Input tensor is wrong size")

        token_positions = torch.arange(0, seq_len, 1)
        # Norm
        inter = self.ln1.forward(x)
        # Multi-head self attention
        inter = self.attn.forward(inter, token_positions)
        # Add
        out = inter + x

        # Norm
        inter = self.ln2.forward(out)
        # Feed-forward (SwiGLU)
        inter = self.ffn.forward(inter)
        # Add
        out = inter + out

        return out


class _TransformerLayers(Module):

    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int,
                 device: torch.device = None, dtype: torch.dtype = None,
                 rope: RotaryPositionalEmbedding = None):
        super().__init__()
        self.num_layers = num_layers
        kwargs = {"device": device, "dtype": dtype, "rope": rope}
        for i in range(num_layers):
            setattr(self, f"{i}", TransformerBlock(d_model, num_heads, d_ff, **kwargs))

    def forward(self, x: torch.Tensor):
        inter = x
        for i in range(self.num_layers):
            tb = getattr(self, f"{i}")
            inter = tb.forward(inter)
        return inter


class TransformerLM(Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int,
                 d_model: int, num_heads: int, d_ff: int, device: torch.device = None,
                 dtype: torch.dtype = None, rope: RotaryPositionalEmbedding = None):
        super().__init__()
        self.num_layers = num_layers
        self.context_length = context_length
        self.kwargs = {"device": device, "dtype": dtype}
        self.token_embeddings = Embedding(num_embeddings=vocab_size,
                                          embedding_dim=d_model,
                                          **self.kwargs)
        self.layers = _TransformerLayers(num_layers, d_model, num_heads, d_ff, **self.kwargs,
                                         rope=rope)
        self.ln_final = RMSNorm(d_model=d_model, **self.kwargs)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, **self.kwargs)

    def forward(self, x: torch.Tensor):
        emb = self.token_embeddings.forward(x)
        out = self.layers.forward(emb)
        out = self.ln_final.forward(out)
        out = self.lm_head.forward(out)
        return out

    def infer(self, x: torch.Tensor):
        with torch.no_grad():
            preds = self.forward(x)
        return preds
