

from __future__ import annotations

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


class NoamScheduler(LRScheduler):
    """Linear warm-up followed by inverse square-root decay."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        self.d_model = d_model
        self.warmup_steps = max(1, int(warmup_steps))
        super().__init__(optimizer, last_epoch)

    def _get_lr_scale(self) -> float:
        # PyTorch's LRScheduler.__init__ calls step() once, which bumps
        # last_epoch from -1 to 0 before invoking get_lr(). Using
        # step = last_epoch + 1 makes the very first reading of
        # optimizer.param_groups[0]['lr'] correspond to step = 1.
        step = max(self.last_epoch + 1, 1)
        d = float(self.d_model)
        w = float(self.warmup_steps)
        return (d ** -0.5) * min(step ** -0.5, step * (w ** -1.5))

    def get_lr(self) -> list[float]:
        scale = self._get_lr_scale()
        return [base_lr * scale for base_lr in self.base_lrs]


def get_lr_history(d_model: int, warmup_steps: int, total_steps: int) -> list[float]:
    """Simulate the Noam LR trajectory for `total_steps` optimisation steps."""
    dummy = torch.nn.Linear(1, 1)
    opt = optim.Adam(dummy.parameters(), lr=1.0)
    sched = NoamScheduler(opt, d_model=d_model, warmup_steps=warmup_steps)

    history: list[float] = []
    for _ in range(total_steps):
        history.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return history


if __name__ == "__main__":  # pragma: no cover
    import matplotlib.pyplot as plt

    D_MODEL = 512
    WARMUP_STEPS = 4000
    TOTAL_STEPS = 20_000

    lrs = get_lr_history(D_MODEL, WARMUP_STEPS, TOTAL_STEPS)

    plt.figure(figsize=(9, 4))
    plt.plot(lrs)
    plt.axvline(WARMUP_STEPS, color="red", linestyle="--", label=f"warmup={WARMUP_STEPS}")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title(f"Noam LR Schedule  (d_model={D_MODEL})")
    plt.legend()
    plt.tight_layout()
    plt.show()
