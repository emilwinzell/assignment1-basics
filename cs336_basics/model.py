import torch
from pathlib import Path
import numpy as np

from .utils import softmax, load_checkpoint
from .train import Hyperparameters, load_hparams_from_file
from .transformer import TransformerLM
from .modules import RotaryPositionalEmbedding
from .optimizer import AdamW
from .tokenizer import Tokenizer


class LanguageModel:

    def __init__(self, tokenizer=None, model: torch.nn.Module = None, max_tokens: int = 1000,
                 temperature: float = 0.2):
        self.tokenizer = tokenizer
        self.endoftext = self.tokenizer.encode("<|endoftext|>")[0]
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def top_k_sampling(probabilities: torch.Tensor, top_k: int = 20):
        return torch.multinomial(probabilities, top_k)

    def _decode_step(self, seq: np.ndarray | list):
        with torch.no_grad():
            preds = self.model.forward(
                torch.tensor(seq)
            )
        q_dist = softmax(preds[:, -1, :], -1, tau=self.temperature)
        return q_dist.flatten()

    def _pre_process_sequence(self, seq):
        # Pad or truncate ?
        pass

    def query(self, query: str) -> str:
        max_seq_len = 100
        base_seq = self.tokenizer.encode(query)
        new_seq = []
        distr = self._decode_step([base_seq])
        next_token = int(torch.multinomial(distr, 1)[0])

        for iter in range(max_seq_len):
            if next_token == self.endoftext:
                break
            new_seq.append(next_token)
            distr = self._decode_step([base_seq + new_seq])
            next_token = int(torch.multinomial(distr, 1)[0])

        return self.tokenizer.decode(new_seq)


def setup_model(path: Path, device: str, vocab_size: int):
    h_params = load_hparams_from_file(path / "config.yaml")
    rope = RotaryPositionalEmbedding(
        theta=h_params.model.rope_theta,
        d_k=h_params.model.d_model // h_params.model.num_heads,
        max_seq_len=h_params.model.context_length,
        device=device
    )
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=h_params.model.context_length,
        num_layers=h_params.model.num_layers,
        d_model=h_params.model.d_model,
        num_heads=h_params.model.num_heads,
        d_ff=h_params.model.d_ff,
        rope=rope,
        device=device
    )
    optimizer = AdamW(
        params=model.parameters(),
        lr=h_params.optimizer.learning_rate,
        betas=h_params.optimizer.betas,
        eps=h_params.optimizer.eps,
        weight_decay=h_params.optimizer.weight_decay
    )
    return model, optimizer


def main():
    # Setup model
    DEVICE = 'cpu'
    VOCAB_SIZE = 10000
    base_dir = Path.cwd()
    model_path = base_dir / "models" / "emil_transformer_tinystories_run_1763114203"
    model, optimizer = setup_model(model_path,
                                   device=DEVICE,
                                   vocab_size=VOCAB_SIZE)
    step = load_checkpoint(model_path / "checkpoint-3416", model, optimizer)

    tokenizer = Tokenizer.from_files(
        vocab_filepath=base_dir / 'data' / 'tinystories_vocab.pkl',
        merges_filepath=base_dir / 'data' / 'tinystories_merges.pkl',
        special_tokens=["<|endoftext|>"]
    )

    decoder = LanguageModel(model=model, tokenizer=tokenizer)
    query = "This is a query for LLM"
    answer = decoder.query(query)
    import pdb; pdb.set_trace()




if __name__=='__main__':
    main()