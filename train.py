
from __future__ import annotations

import argparse
import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in Vaswani et al. (2017), sec 5.4.

        y_smooth = (1 - eps) * one_hot(y) + eps / (V - 1)

    The <pad> column receives 0 probability mass and PAD targets are
    ignored when averaging.
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        # Distribute smoothing mass over (V - 2): exclude correct + pad.
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B*T, V]; target: [B*T]
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / max(self.vocab_size - 2, 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_idx] = 0.0
            mask = target == self.pad_idx
            true_dist[mask] = 0.0

        loss = -(true_dist * log_probs).sum(dim=1)
        # Average only over non-pad positions for a numerically meaningful loss.
        n_valid = (~mask).sum().clamp(min=1)
        return loss.sum() / n_valid


# ══════════════════════════════════════════════════════════════════════
#  TRAINING / EVAL LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    pad_idx: int = 1,
    grad_clip: Optional[float] = 1.0,
    log_grad_norms: bool = False,
    grad_norm_layer: Optional[nn.Module] = None,
    use_wandb: bool = False,
    log_step_lr: bool = False,
) -> float:
    """One pass over `data_iter`. Returns avg per-token loss."""
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0
    grad_norms_q: list[float] = []
    grad_norms_k: list[float] = []

    if use_wandb:
        try:
            import wandb  # noqa: F401
        except Exception:
            use_wandb = False

    for batch_idx, (src, tgt) in enumerate(data_iter):
        src = src.to(device)
        tgt = tgt.to(device)

        # Shift target: input drops <eos>, gold drops <sos>.
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = make_src_mask(src, pad_idx)
        tgt_mask = make_tgt_mask(tgt_in, pad_idx)

        logits = model(src, tgt_in, src_mask, tgt_mask)  # [B, T-1, V]
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        if is_train:
            optimizer.zero_grad()
            loss.backward()

            if log_grad_norms and grad_norm_layer is not None:
                # Layer should be a MultiHeadAttention module.
                if grad_norm_layer.W_q.weight.grad is not None:
                    grad_norms_q.append(grad_norm_layer.W_q.weight.grad.norm().item())
                if grad_norm_layer.W_k.weight.grad is not None:
                    grad_norms_k.append(grad_norm_layer.W_k.weight.grad.norm().item())

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if use_wandb and log_step_lr:
                import wandb
                wandb.log({
                    "lr": optimizer.param_groups[0]["lr"],
                    "train/step_loss": loss.item(),
                })

        n_tokens = (tgt_out != pad_idx).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)

    if log_grad_norms and use_wandb and grad_norms_q:
        import wandb
        for i, (gq, gk) in enumerate(zip(grad_norms_q, grad_norms_k)):
            wandb.log({"grad_norm/W_q": gq, "grad_norm/W_k": gk, "step_in_window": i})

    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: Optional[int] = None,
    device: str = "cpu",
    pad_idx: int = 1,
) -> torch.Tensor:
    """Greedy autoregressive decoding. Returns indices [1, out_len]."""
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx)
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if end_symbol is not None and next_token.item() == end_symbol:
                break
    return ys


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def _ids_to_tokens(ids, vocab, sos_idx, eos_idx, pad_idx):
    tokens = []
    for i in ids:
        i = int(i)
        if i == sos_idx:
            continue
        if i == eos_idx or i == pad_idx:
            break
        tokens.append(vocab.itos[i])
    return tokens


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
    pad_idx: int = 1,
) -> float:
    """Corpus-level BLEU (range 0-100) via sacrebleu if available, else nltk."""
    model.eval()

    sos_idx = tgt_vocab.stoi["<sos>"]
    eos_idx = tgt_vocab.stoi["<eos>"]

    hyps: list[list[str]] = []
    refs: list[list[str]] = []

    for src, tgt in test_dataloader:
        src = src.to(device)
        for i in range(src.size(0)):
            src_i = src[i : i + 1]
            src_mask = make_src_mask(src_i, pad_idx)
            out = greedy_decode(
                model, src_i, src_mask, max_len, sos_idx, eos_idx, device, pad_idx
            )
            hyp = _ids_to_tokens(out.squeeze(0).tolist(), tgt_vocab, sos_idx, eos_idx, pad_idx)
            ref = _ids_to_tokens(tgt[i].tolist(), tgt_vocab, sos_idx, eos_idx, pad_idx)
            hyps.append(hyp)
            refs.append(ref)

    return _bleu_score(hyps, refs)


def _bleu_score(hyps: list[list[str]], refs: list[list[str]]) -> float:
    # Try sacrebleu first (robust, normalised tokenisation).
    try:
        import sacrebleu

        hyp_str = [" ".join(h) for h in hyps]
        ref_str = [" ".join(r) for r in refs]
        return float(sacrebleu.corpus_bleu(hyp_str, [ref_str]).score)
    except Exception:
        pass
    # Fallback: nltk corpus BLEU.
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

        smoothing = SmoothingFunction().method1
        return 100.0 * corpus_bleu(
            [[r] for r in refs], hyps, smoothing_function=smoothing
        )
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
    src_vocab=None,
    tgt_vocab=None,
    src_spacy_model: str = "de_core_news_sm",
) -> None:
    model_config = {
        "src_vocab_size": model.src_vocab_size,
        "tgt_vocab_size": model.tgt_vocab_size,
        "d_model": model.d_model,
        "N": model.N,
        "num_heads": model.num_heads,
        "d_ff": model.d_ff,
        "dropout": model.dropout,
        "pad_idx": model.pad_idx,
        "positional_encoding": model.positional_encoding,
    }
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_config": model_config,
        "src_spacy_model": src_spacy_model,
    }
    if src_vocab is not None:
        payload["src_vocab"] = {"itos": list(src_vocab.itos)}
    if tgt_vocab is not None:
        payload["tgt_vocab"] = {"itos": list(tgt_vocab.itos)}
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    return int(state.get("epoch", 0))


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment(args=None) -> None:
    """End-to-end training driver for all five W&B experiments."""
    if args is None:
        args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ─ W&B ───────────────────────────────────────────────────────────
    use_wandb = args.use_wandb
    if use_wandb:
        try:
            import wandb

            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args),
            )
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] wandb unavailable ({exc}); disabling.")
            use_wandb = False

    # ─ Data ──────────────────────────────────────────────────────────
    from dataset import make_dataloaders, PAD_IDX

    pack = make_dataloaders(batch_size=args.batch_size, min_freq=args.min_freq)
    train_loader = pack["train_loader"]
    val_loader = pack["val_loader"]
    test_loader = pack["test_loader"]
    src_vocab = pack["src_vocab"]
    tgt_vocab = pack["tgt_vocab"]
    pad_idx = pack["pad_idx"]

    print(f"|src vocab| = {len(src_vocab)}   |tgt vocab| = {len(tgt_vocab)}")

    # ─ Model ─────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        N=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pad_idx=pad_idx,
        positional_encoding=args.positional_encoding,
    ).to(device)
    model.attach_vocab(src_vocab, tgt_vocab, pack["src_tokenizer"])

    # ─ Disable scaling factor (Section 2.2 ablation) ────────────────
    if args.disable_scale:
        for m in model.modules():
            from model import MultiHeadAttention

            if isinstance(m, MultiHeadAttention):
                m.use_scale = False

    # ─ Optimiser ─────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.base_lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    # ─ Scheduler ─────────────────────────────────────────────────────
    if args.scheduler == "noam":
        from lr_scheduler import NoamScheduler

        scheduler = NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps)
    else:
        scheduler = None

    # ─ Loss ──────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=pad_idx,
        smoothing=args.label_smoothing,
    )

    # ─ Train ─────────────────────────────────────────────────────────
    best_val = math.inf
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_path = os.path.join(args.ckpt_dir, f"{args.run_name}_best.pt")

    grad_norm_layer = (
        model.encoder.layers[0].self_attn if args.log_grad_norms else None
    )

    for epoch in range(args.num_epochs):
        t0 = time.time()
        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device,
            pad_idx=pad_idx,
            log_grad_norms=args.log_grad_norms and epoch == 0,
            grad_norm_layer=grad_norm_layer,
            use_wandb=use_wandb,
            log_step_lr=True,
        )
        val_loss = run_epoch(
            val_loader,
            model,
            loss_fn,
            None,
            None,
            epoch_num=epoch,
            is_train=False,
            device=device,
            pad_idx=pad_idx,
        )
        elapsed = time.time() - t0
        val_ppl = math.exp(min(val_loss, 20))

        print(
            f"[epoch {epoch+1:02d}/{args.num_epochs}] "
            f"train={train_loss:.3f}  val={val_loss:.3f}  ppl={val_ppl:.2f}  ({elapsed:.0f}s)"
        )

        if use_wandb:
            import wandb
            log = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/perplexity": val_ppl,
            }
            if args.log_prediction_confidence:
                log["val/prediction_confidence"] = compute_prediction_confidence(
                    model, val_loader, device, pad_idx
                )
            wandb.log(log)

        # Always save the latest, and track the best.
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            os.path.join(args.ckpt_dir, f"{args.run_name}_last.pt"),
            src_vocab=src_vocab, tgt_vocab=tgt_vocab,
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_path,
                src_vocab=src_vocab, tgt_vocab=tgt_vocab,
            )

    # ─ Final BLEU on test ────────────────────────────────────────────
    if os.path.exists(best_path):
        load_checkpoint(best_path, model)
    bleu = evaluate_bleu(
        model, test_loader, tgt_vocab, device=device, pad_idx=pad_idx
    )
    print(f"Test BLEU = {bleu:.2f}")
    if use_wandb:
        import wandb
        wandb.log({"test/bleu": bleu})
        wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#  HELPERS for the W&B report
# ══════════════════════════════════════════════════════════════════════

def compute_prediction_confidence(
    model: Transformer,
    data_loader: DataLoader,
    device: str,
    pad_idx: int,
) -> float:
    """Mean softmax probability assigned to the correct token (val set)."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            src_mask = make_src_mask(src, pad_idx)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx)
            logits = model(src, tgt_in, src_mask, tgt_mask)  # [B, T-1, V]
            probs = F.softmax(logits, dim=-1)
            gather = probs.gather(2, tgt_out.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
            mask = tgt_out != pad_idx
            total += gather[mask].sum().item()
            n += mask.sum().item()
    return total / max(n, 1)


# ══════════════════════════════════════════════════════════════════════
#  ARGS
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    # Architecture
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--positional_encoding", choices=["sinusoidal", "learned"], default="sinusoidal"
    )
    p.add_argument("--disable_scale", action="store_true",
                   help="Disable 1/sqrt(d_k) (Section 2.2 ablation).")
    # Training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=20)
    p.add_argument("--min_freq", type=int, default=2)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--scheduler", choices=["noam", "fixed"], default="noam")
    p.add_argument("--warmup_steps", type=int, default=4000)
    p.add_argument("--base_lr", type=float, default=1.0,
                   help="With Noam, base_lr=1.0 lets the schedule control LR. "
                        "With fixed, e.g. 1e-4.")
    # Logging
    p.add_argument("--log_grad_norms", action="store_true")
    p.add_argument("--log_prediction_confidence", action="store_true")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="da6401-a3")
    p.add_argument("--run_name", default="baseline")
    p.add_argument("--ckpt_dir", default="checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    run_training_experiment()
