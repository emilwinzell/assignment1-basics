import torch
import numpy as np
import typing
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from functools import partial

from .transformer import TransformerLM
from .optimizer import AdamW
from .modules import RotaryPositionalEmbedding
from .utils import (get_batch, cosine_annealing_lr_schedule, gradient_clipping,
                    cross_entropy_loss, save_checkpoint)


@dataclass
class Hyperparameters:
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    learning_rate: int
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0
    rope_theta: float = 10000
    cos_an_lr_max: float = 1.0
    cos_an_lr_min: float = 1e-6
    cos_an_warm_up: int = 50000
    cos_an_annealing: int = 1e6
    grad_clip_max_norm: float = 0.01


class Trainer:

    def __init__(self, model: torch.nn.Module, optimizer: torch.nn.Module, loss: typing.Callable,
                 batch_size: int, num_epochs: int, learning_rate: float = 1e-3,
                 eval_steps: int = 1,
                 data_loader: typing.Callable | None = None,
                 learning_rate_schedule: typing.Callable | None = None,
                 use_gradient_clipping: bool = True,
                 grad_clip_args: tuple | None = None,
                 checkpointing: bool = True,
                 save_path: str | Path = None,
                 device: str = None,
                 ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss
        self.lr_schedule = learning_rate_schedule
        self.device = device
        self.train_args = {"batch_size": batch_size,
                           "context_length": self.model.context_length,
                           "num_epochs": num_epochs,
                           "learning_rate": learning_rate,
                           "use_grad_clip": use_gradient_clipping,
                           "grad_clip_args": grad_clip_args,
                           "checkpointing": checkpointing,
                           "eval_steps": eval_steps,
                           "save_path": Path.cwd() if save_path is None else save_path}
        if data_loader is not None:
            self.data_loader = data_loader
        else:
            self.data_loader = partial(self._simple_data_load,
                                       context_length=self.train_args["context_length"])

        self._last_lr: float = 0.0
        self._last_step: int = 0
        self.train_log: list = []

    def _update_lr(self):
        step = self._last_step + 1

        if self.lr_schedule is None:
            if step > 1:
                self._last_step = step
                return  # No need to update again
            lr = self.train_args["learning_rate"]
        else:
            lr = self.lr_schedule(step)

        for param_group in self.optimizer.param_groups:
            if isinstance(param_group["lr"], torch.Tensor):
                param_group["lr"].fill_(lr)
            else:
                param_group["lr"] = lr

        self._last_lr = lr
        self._last_step = step

    def _train_step(self, x: torch.Tensor, y: torch.Tensor):

        # Update learning rate
        self._update_lr()

        # Run model
        preds = self.model.forward(x)

        self.optimizer.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = self.loss_fn(preds, y)
        loss.backward()  # Run backward pass, which computes gradients.

        if self.train_args["use_grad_clip"]:
            gradient_clipping(self.model.parameters(), *self.train_args["grad_clip_args"])

        self.optimizer.step()  # Run optimizer step.

        return loss

    def _validation_step(self, validation_data: np.ndarray):
        x, y = self._simple_data_load(validation_data,
                                      context_length=self.train_args["context_length"])

        with torch.no_grad():
            # Run inference
            preds = self.model.forward(x)
            return self.loss_fn(preds, y)

    def _simple_data_load(self, x: np.ndarray, context_length: int
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0] - context_length - 1  # Grab whole array as one batch
        return get_batch(x, batch_size, context_length, self.device)

    def train(self, train_data: np.ndarray, validation_data: np.ndarray):
        num_epochs = self.train_args["num_epochs"]
        batch_size = self.train_args["batch_size"]
        num_steps_per_epoch = train_data.shape[0] // batch_size

        for step in tqdm(range(num_steps_per_epoch * num_epochs)):
            current_epoch = step / num_steps_per_epoch

            x, y = self.data_loader(train_data)

            tr_loss = self._train_step(x, y)

            if step % self.train_args["eval_steps"] == 0:
                val_loss = self._validation_step(validation_data)
                self.train_log.append(
                    {"step": step, "epoch": current_epoch, "train_loss": tr_loss,
                     "val_loss": val_loss}
                )
                if self.train_args["checkpointing"]:
                    best_loss = min([log["val_loss"] for log in self.train_log])
                    if val_loss < best_loss:
                        print(f"Checkpoint saved at step {step}, epoch {current_epoch}, "
                              f"validation loss={val_loss}")
                        save_checkpoint(self.model, self.optimizer, iteration=step,
                                        out=self.train_args["save_path"])

        best_loss = min([log["val_loss"] for log in self.train_log])
        print(f"Training complete, best loss: {best_loss}")


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, mmap_mode='r') as data:
        train = data['train']
        valid = data['valid']
    return train, valid


def main():
    DATASET = 'tinystories'
    SAVE_NAME = f'model_{DATASET}_run_0'
    BATCH_SIZE = 4
    NUM_EPOCHS = 3
    DEVICE = 'cpu'

    h_params = Hyperparameters(
        context_length=512,
        num_layers=2,
        d_model=768,
        num_heads=4,
        d_ff=320,
        learning_rate=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-6,
        rope_theta=10000,
        cos_an_lr_max=1,
        cos_an_lr_min=1e-6,
        cos_an_warm_up=50000,
        cos_an_annealing=1e6,
        grad_clip_max_norm=0.01,
    )

    print("loading..")
    train, valid = load_dataset(path=Path.cwd() / "data" / f"{DATASET}_encoded.npz")
    vocab_size = np.max(train) + 1

    rope = RotaryPositionalEmbedding(theta=h_params.rope_theta,
                                     d_k=h_params.d_model // h_params.num_heads,
                                     max_seq_len=h_params.context_length)
    model = TransformerLM(vocab_size=vocab_size,
                          context_length=h_params.context_length,
                          num_layers=h_params.num_layers,
                          d_model=h_params.d_model,
                          num_heads=h_params.num_heads,
                          d_ff=h_params.d_ff,
                          rope=rope)
    optimizer = AdamW(params=model.parameters(),
                      lr=h_params.learning_rate,
                      betas=h_params.betas,
                      eps=h_params.eps,
                      weight_decay=h_params.weight_decay)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss=cross_entropy_loss,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        eval_steps=20,
        data_loader=partial(get_batch, batch_size=BATCH_SIZE,
                            context_length=h_params.context_length, device=DEVICE),
        learning_rate_schedule=partial(cosine_annealing_lr_schedule, lr_max=h_params.cos_an_lr_max,
                                       lr_min=h_params.cos_an_lr_min,
                                       warm_up=h_params.cos_an_warm_up,
                                       annealing=h_params.cos_an_annealing),
        grad_clip_args=(h_params.grad_clip_max_norm,),
        save_path=Path.cwd() / SAVE_NAME,
        device=DEVICE
    )
    import pdb; pdb.set_trace()
    trainer.train(train, valid)


if __name__ == '__main__':
    main()
