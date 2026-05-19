

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Sequence, Tuple, Optional

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


SPECIAL_TOKENS = ("<unk>", "<pad>", "<sos>", "<eos>")
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3


# ─────────────────────────────────────────────────────────────────────
#  Vocab
# ─────────────────────────────────────────────────────────────────────

class Vocab:
    """Minimal torchtext-compatible vocabulary."""

    def __init__(self, tokens: Sequence[str]) -> None:
        self.itos: List[str] = list(tokens)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def lookup_token(self, idx: int) -> str:
        return self.itos[idx]

    def encode(self, tokens: Iterable[str]) -> List[int]:
        unk = self.stoi["<unk>"]
        return [self.stoi.get(t, unk) for t in tokens]


def _build_vocab(token_iter: Iterable[List[str]], min_freq: int = 2) -> Vocab:
    counter: Counter = Counter()
    for toks in token_iter:
        counter.update(toks)
    itos = list(SPECIAL_TOKENS)
    for tok, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if freq >= min_freq and tok not in SPECIAL_TOKENS:
            itos.append(tok)
    return Vocab(itos)


# ─────────────────────────────────────────────────────────────────────
#  Multi30k dataset
# ─────────────────────────────────────────────────────────────────────

class Multi30kDataset(Dataset):
    """
    bentrevett/multi30k German -> English.

    Each item is a tuple (src_ids, tgt_ids) of LongTensor 1-D index
    sequences with <sos>/<eos> wrapped, ready for collation.
    """

    def __init__(
        self,
        split: str = "train",
        src_vocab: Optional[Vocab] = None,
        tgt_vocab: Optional[Vocab] = None,
        src_tokenizer=None,
        tgt_tokenizer=None,
        min_freq: int = 2,
        lowercase: bool = True,
    ) -> None:
        from datasets import load_dataset
        import spacy

        self.split = split
        self.lowercase = lowercase

        ds = load_dataset("bentrevett/multi30k", split=split)
        self.raw = [(ex["de"], ex["en"]) for ex in ds]

        # Lazy-load spaCy models; users must install them once:
        #   python -m spacy download de_core_news_sm
        #   python -m spacy download en_core_web_sm
        self.src_tokenizer = src_tokenizer or spacy.load(
            "de_core_news_sm", disable=["tagger", "parser", "ner", "lemmatizer"]
        )
        self.tgt_tokenizer = tgt_tokenizer or spacy.load(
            "en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"]
        )

        self.src_tokens = [self._tokenize(self.src_tokenizer, s) for s, _ in self.raw]
        self.tgt_tokens = [self._tokenize(self.tgt_tokenizer, t) for _, t in self.raw]

        # Build vocab from training split only; reuse for val/test.
        if src_vocab is None or tgt_vocab is None:
            assert split == "train", "Pass src_vocab/tgt_vocab for non-train splits."
            self.src_vocab = _build_vocab(self.src_tokens, min_freq=min_freq)
            self.tgt_vocab = _build_vocab(self.tgt_tokens, min_freq=min_freq)
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

    def _tokenize(self, nlp, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        return [t.text for t in nlp(text)]

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_ids = (
            [self.src_vocab.stoi["<sos>"]]
            + self.src_vocab.encode(self.src_tokens[idx])
            + [self.src_vocab.stoi["<eos>"]]
        )
        tgt_ids = (
            [self.tgt_vocab.stoi["<sos>"]]
            + self.tgt_vocab.encode(self.tgt_tokens[idx])
            + [self.tgt_vocab.stoi["<eos>"]]
        )
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

    # Optional: keep API shape requested by the skeleton.
    def build_vocab(self):  # pragma: no cover - vocabularies built in __init__
        return self.src_vocab, self.tgt_vocab

    def process_data(self):  # pragma: no cover
        return self.src_tokens, self.tgt_tokens


# ─────────────────────────────────────────────────────────────────────
#  Collate
# ─────────────────────────────────────────────────────────────────────

def collate_fn(batch, pad_idx: int = PAD_IDX):
    src_seqs, tgt_seqs = zip(*batch)
    src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_seqs, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded


def make_dataloaders(
    batch_size: int = 64,
    min_freq: int = 2,
    num_workers: int = 0,
):
    """Build train/val/test DataLoaders sharing the train split's vocab."""
    from torch.utils.data import DataLoader

    train_ds = Multi30kDataset(split="train", min_freq=min_freq)
    val_ds = Multi30kDataset(
        split="validation",
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab,
        src_tokenizer=train_ds.src_tokenizer,
        tgt_tokenizer=train_ds.tgt_tokenizer,
    )
    test_ds = Multi30kDataset(
        split="test",
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab,
        src_tokenizer=train_ds.src_tokenizer,
        tgt_tokenizer=train_ds.tgt_tokenizer,
    )

    pad_idx = train_ds.src_vocab.stoi["<pad>"]
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx),
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx),
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx),
        num_workers=num_workers,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "src_vocab": train_ds.src_vocab,
        "tgt_vocab": train_ds.tgt_vocab,
        "src_tokenizer": train_ds.src_tokenizer,
        "tgt_tokenizer": train_ds.tgt_tokenizer,
        "pad_idx": pad_idx,
    }
