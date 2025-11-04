from collections.abc import Callable
import torch
import math


class SGD(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss


class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-08, weight_decay: float = 0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay}
        super().__init__(params, defaults)
        self._init_states()

    def _init_states(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["t"] = torch.tensor(1, dtype=p.data.dtype)
                state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def step(self, closure: Callable | None = None):
        """Take one step of Adam with weight decay."""
        loss = None if closure is None else closure()

        m: torch.Tensor
        v: torch.Tensor
        grad: torch.Tensor
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            b1, b2 = group["betas"]
            eps = group["eps"]
            wdec = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state["t"]  # Get iteration number from the state, or initial value.
                m, v = state["m"], state["v"]  # Get the state vectors
                grad = p.grad.data

                # Update state vectors
                m = m.lerp(grad, 1 - b1)
                v = v.lerp(grad.square(), 1 - b2)
                # Compute adjusted lr for iteration t
                lr_t = lr * math.sqrt(1 - b2 ** t) / (1 - b1 ** t)
                p.data = p.data.addcdiv(m, v.sqrt().add(eps), value=-lr_t)
                p.data = p.data.add(alpha=-lr * wdec, other=p.data)  # Apply weight decay
                # Update iteration number and vectors
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss


def main():
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e-2)

    for _ in range(100):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.


if __name__ == '__main__':
    main()
