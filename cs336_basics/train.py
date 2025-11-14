import os
import torch
import numpy as np
import typing
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass, is_dataclass, fields, asdict
from functools import partial
import wandb
import yaml
import argparse
import time
import shutil

from .transformer import TransformerLM
from .optimizer import AdamW
from .modules import RotaryPositionalEmbedding
from .utils import (get_batch, cosine_annealing_lr_schedule, gradient_clipping,
                    cross_entropy_loss, save_checkpoint)

os.environ['WANDB_API_KEY'] = 'c052c7a425c0ab54ef937eea1b15604fba8f20a1'


@dataclass
class OptimizerParams:
    learning_rate: int
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0


@dataclass
class ModelParams:
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    rope_theta: float = 10000


@dataclass
class TrainParams:
    batch_size: int
    total_token_proc: int = -1
    num_steps: int = -1
    eval_interval: int = -1
    eval_steps: int = -1
    # LR schedule
    cos_an_lr_max: float = 1.0
    cos_an_lr_min: float = 1e-6
    cos_an_warm_up: int = 0
    cos_an_annealing: int = 0
    # Gradient clipping
    grad_clip_max_norm: float = 0.01

    def __post_init__(self):
        if self.num_steps == -1 and self.total_token_proc == -1:
            raise ValueError("Need to specify either steps or total tokens processed!")


@dataclass
class Hyperparameters:
    name: str
    train: TrainParams
    model: ModelParams
    optimizer: OptimizerParams

    def __post_init__(self):
        if self.train.total_token_proc == -1:
            self.train.total_token_proc = (self.model.context_length * self.train.num_steps *
                                           self.train.batch_size)
        elif self.train.num_steps == -1:
            self.train.num_steps = int(self.train.total_token_proc /
                                       (self.model.context_length * self.train.batch_size))
        self.train.cos_an_warm_up = int(self.train.num_steps * 0.03)  # 3% of total steps
        self.train.cos_an_annealing = self.train.num_steps + 1  # Run until end
        if self.train.eval_interval == -1:
            self.train.eval_interval = int(self.train.num_steps * 0.05)  # Validate every 5%

    def save(self, path: str | Path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)


def load_hparams_from_file(config_path: str | Path) -> Hyperparameters:
    """Load configuration from a file and return as class"""

    def _load_from_dict(d_class, data: dict):
        kwargs = {}
        for field in fields(d_class):
            if field.name in data:
                value = data[field.name]

                if isinstance(field.type, tuple) and isinstance(value, list):
                    value = tuple(value)

                # Handle nested dataclasses and lists
                if is_dataclass(field.type):
                    kwargs[field.name] = _load_from_dict(field.type, value)
                else:
                    kwargs[field.name] = value
        return d_class(**kwargs)

    if isinstance(config_path, str):
        config_path = Path(config_path)

    with open(config_path) as f:
        if config_path.suffix in [".yaml", ".yml"]:
            raw_data = yaml.safe_load(f)
        else:
            return None

    return _load_from_dict(Hyperparameters, raw_data)


class Logger:

    def __init__(self, save_path: Path, project: str):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        name = save_path.stem
        id = name.split("_")[-1]
        self.wandb_run = wandb.init(
            project=project,
            id=id if id else '00000',
            dir=save_path
        )

    def log(self, stuff: dict):
        self.wandb_run.log(stuff)


class Trainer:

    def __init__(self, model: torch.nn.Module, optimizer: torch.nn.Module, loss: typing.Callable,
                 batch_size: int,
                 num_epochs: int = -1, num_steps: int = -1,
                 learning_rate: float = 1e-3,
                 eval_steps: int = 1, eval_interval: int = 1,
                 data_loader: typing.Callable | None = None,
                 learning_rate_schedule: typing.Callable | None = None,
                 use_gradient_clipping: bool = True,
                 grad_clip_args: tuple | None = None,
                 checkpointing: bool = True,
                 save_path: Path = None,
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
                           "num_steps": num_steps if num_epochs == -1 else -1,
                           "learning_rate": learning_rate,
                           "use_grad_clip": use_gradient_clipping,
                           "grad_clip_args": grad_clip_args,
                           "checkpointing": checkpointing,
                           "eval_steps": eval_steps,
                           "eval_interval": eval_interval,
                           "save_path": Path.cwd() if save_path is None else save_path}
        if data_loader is not None:
            self.data_loader = data_loader
        else:
            self.data_loader = partial(self._simple_data_load,
                                       context_length=self.train_args["context_length"])

        self._last_lr: float = 0.0
        self._last_step: int = 0
        self._best_val_loss: float = np.inf
        self.logger = Logger(save_path, "stanford-cs336_basics")

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

        st_time = time.time()
        log_metrics = {}
        # Update learning rate
        self._update_lr()
        log_metrics["train/lr"] = self._last_lr

        # Run model
        preds = self.model.forward(x)

        self.optimizer.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = self.loss_fn(preds, y)
        loss_out = loss.detach()
        loss.backward()  # Run backward pass, which computes gradients.

        if self.train_args["use_grad_clip"]:
            norm = gradient_clipping(self.model.parameters(), *self.train_args["grad_clip_args"])

        self.optimizer.step()  # Run optimizer step.

        log_metrics["train/loss"] = loss_out
        log_metrics["train/gradient_norm"] = norm
        log_metrics["train/tokens_per_sec"] = (
            self.train_args["batch_size"] * self.train_args["context_length"] /
            (time.time() - st_time)
        )

        return log_metrics

    def _validation_step(self, validation_data: np.ndarray):
        self.model.eval()
        num_steps = self.train_args["eval_steps"]
        batch_loss = 0.0
        for _ in range(num_steps):
            x, y = self.data_loader(validation_data)
            with torch.no_grad():
                # Run inference
                preds = self.model.forward(x)
                loss = self.loss_fn(preds, y)
                batch_loss += loss.item()
        self.model.train()
        return batch_loss / num_steps

    def _simple_data_load(self, x: np.ndarray, context_length: int
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0] - context_length - 1  # Grab whole array as one batch
        return get_batch(x, batch_size, context_length, self.device)

    def _save_checkpoint(self, model, optimizer, iteration, out):
        max_checkpoints = 3
        save_checkpoint(model, optimizer, iteration, out)
        chpts = [ch for ch in self.train_args["save_path"].rglob("checkpoint-*")]
        if len(chpts) > max_checkpoints:
            checkpt = chpts[0]
            try:
                shutil.rmtree(checkpt)
            except OSError as e:
                print(f"Error in removing old checkpoint: {e.filename} - {e.strerror}.")

    def train(self, train_data: np.ndarray, validation_data: np.ndarray):
        num_epochs = self.train_args["num_epochs"]
        batch_size = self.train_args["batch_size"]
        num_steps_per_epoch = train_data.shape[0] // batch_size

        num_steps = self.train_args["num_steps"]
        if num_steps == -1:
            num_steps = num_steps_per_epoch * num_epochs

        self.model.train()

        for step in tqdm(range(num_steps)):
            current_epoch = step / num_steps_per_epoch

            x, y = self.data_loader(train_data)

            metrics = self._train_step(x, y)
            metrics["step"] = step
            self.logger.log(metrics)

            if step % self.train_args["eval_interval"] == 0:
                val_loss = self._validation_step(validation_data)
                self.logger.log(
                    {"vallidation/loss": val_loss, "step": step}
                )

                if self.train_args["checkpointing"]:
                    if val_loss <= self._best_val_loss:
                        print(f"Checkpoint saved at step {step}, epoch {current_epoch}, "
                              f"validation loss={val_loss}")
                        self._save_checkpoint(
                            self.model, self.optimizer, iteration=step,
                            out=self.train_args["save_path"] / f"checkpoint-{step}"
                        )

                        self._best_val_loss = val_loss

        print(f"Training complete, best loss: {self._best_val_loss}")


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, mmap_mode='r') as data:
        train = data['train']
        valid = data['valid']
    return train, valid


def main(args):
    h_params = load_hparams_from_file(
        args.config if args.config else Path.cwd() / "config" / "tiny.yaml"
    )

    DATASET = 'tinystories'
    TIMESTAMP = int(time.time())
    SAVE_NAME = f'{h_params.name}_{DATASET}_run_{TIMESTAMP}'
    SAVE_PATH = Path.cwd() / "models" / SAVE_NAME
    DEVICE = 'cpu'

    print("loading..")
    train, valid = load_dataset(path=Path.cwd() / "data" / f"{DATASET}_encoded.npz")
    vocab_size = np.max(train) + 1

    rope = RotaryPositionalEmbedding(
        theta=h_params.model.rope_theta,
        d_k=h_params.model.d_model // h_params.model.num_heads,
        max_seq_len=h_params.model.context_length,
        device=DEVICE
    )
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=h_params.model.context_length,
        num_layers=h_params.model.num_layers,
        d_model=h_params.model.d_model,
        num_heads=h_params.model.num_heads,
        d_ff=h_params.model.d_ff,
        rope=rope,
        device=DEVICE
    )
    optimizer = AdamW(
        params=model.parameters(),
        lr=h_params.optimizer.learning_rate,
        betas=h_params.optimizer.betas,
        eps=h_params.optimizer.eps,
        weight_decay=h_params.optimizer.weight_decay
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss=cross_entropy_loss,
        batch_size=h_params.train.batch_size,
        num_steps=h_params.train.num_steps,
        eval_steps=h_params.train.eval_steps,
        eval_interval=h_params.train.eval_interval,
        data_loader=partial(get_batch, batch_size=h_params.train.batch_size,
                            context_length=h_params.model.context_length, device=DEVICE),
        learning_rate_schedule=partial(cosine_annealing_lr_schedule,
                                       lr_max=h_params.train.cos_an_lr_max,
                                       lr_min=h_params.train.cos_an_lr_min,
                                       warm_up=h_params.train.cos_an_warm_up,
                                       annealing=h_params.train.cos_an_annealing),
        grad_clip_args=(h_params.train.grad_clip_max_norm,),
        save_path=SAVE_PATH,
        device=DEVICE
    )

    trainer.train(train, valid)
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a transformer model")
    parser.add_argument("--config", type=str, help="Path to config file (optional)")
    #parser.add_argument("--resume-from", type=str, help="Path to run directory to resume from")
    args = parser.parse_args()

    main(args)
