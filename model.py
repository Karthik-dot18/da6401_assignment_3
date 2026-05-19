

import math
import copy
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gdown  # noqa: F401
except Exception:  # pragma: no cover - optional dep
    gdown = None


# ══════════════════════════════════════════════════════════════════════
#  AUTOGRADER ENTRY-POINT CONFIG
#
#  The grader calls `Transformer()` with no arguments and then
#  `model.infer(<german sentence>)`. To support that, this file:
#    1. Treats *every* __init__ arg as optional.
#    2. Downloads a pre-trained checkpoint from Google Drive when no
#       weights are otherwise supplied. Replace GDRIVE_CHECKPOINT_ID
#       with the share-ID of your trained .pt file before submitting.
#    3. Reconstructs the architecture + vocab from the checkpoint so
#       that `infer()` works end-to-end with no setup call.
#
#  Get the ID from the share link e.g. drive.google.com/file/d/<ID>/view
# ══════════════════════════════════════════════════════════════════════
GDRIVE_CHECKPOINT_ID = "1SNXLGNTvDWtXSyWGlJdO8XP90lT-4hWM"
DEFAULT_CHECKPOINT_PATH = "checkpoint.pt"


# ══════════════════════════════════════════════════════════════════════
#  STANDALONE ATTENTION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax(Q . K^T / sqrt(d_k)) . V

    Args:
        Q    : (..., seq_q, d_k)
        K    : (..., seq_k, d_k)
        V    : (..., seq_k, d_v)
        mask : Boolean mask broadcastable to (..., seq_q, seq_k).
               True positions are MASKED OUT.
        scale: If False, omits the 1/sqrt(d_k) scaling factor (used for
               the ablation study in Section 2.2 of the report).

    Returns:
        output : (..., seq_q, d_v)
        attn_w : (..., seq_q, seq_k)  -- softmax-normalised weights
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if scale:
        scores = scores / math.sqrt(d_k)

    if mask is not None:
        # mask True -> position is invalid -> set to very negative number
        scores = scores.masked_fill(mask, float("-inf"))

    attn_w = F.softmax(scores, dim=-1)
    # Guard against rows that were entirely masked (all -inf -> NaN softmax).
    attn_w = torch.nan_to_num(attn_w, nan=0.0)
    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Padding mask for the encoder.

    Args:
        src     : [batch, src_len]
        pad_idx : <pad> index (default 1).

    Returns:
        Bool tensor [batch, 1, 1, src_len].
        True  -> PAD (masked out).
        False -> real token.
    """
    # src == pad_idx -> True (masked); insert head & query dims for broadcast.
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Combined padding + look-ahead (causal) mask for the decoder.

    Args:
        tgt     : [batch, tgt_len]
        pad_idx : <pad> index (default 1).

    Returns:
        Bool tensor [batch, 1, tgt_len, tgt_len].
        True -> masked out (PAD or future token).
    """
    batch, tgt_len = tgt.shape
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)              # [B,1,1,T]
    causal = torch.triu(
        torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=tgt.device),
        diagonal=1,
    )                                                                  # [T,T]
    causal = causal.unsqueeze(0).unsqueeze(0)                          # [1,1,T,T]
    return pad_mask | causal                                           # [B,1,T,T]


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention (Vaswani et al., 2017, sec 3.2.2)."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Toggle for the 1/sqrt(d_k) ablation (Section 2.2 of the report).
        self.use_scale = True
        # Cache attention weights from the last forward pass for visualisation.
        self.last_attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, D] -> [B, H, L, d_k]"""
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, L, d_k] -> [B, L, D]"""
        b, _, l, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, l, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[dict] = None,
        cache_kind: str = "none",  # 'none' | 'self' | 'cross'
    ) -> torch.Tensor:
        """Standard MHA, with an optional KV-cache for fast greedy decode.

        cache_kind:
          - 'none'  : no caching (training, full forward).
          - 'self'  : decoder self-attention; caller passes only the *new*
                      query/key/value tokens (typically 1 step at a time).
                      We append to cache['k']/cache['v'] and attend over
                      the full history, ignoring `mask` (causality is
                      automatic because we never see future tokens).
          - 'cross' : decoder cross-attention; key/value come from
                      encoder memory and are constant across decode
                      steps. We compute them once, store them in cache,
                      and reuse on subsequent calls.
        """
        Q = self._split_heads(self.W_q(query))

        if cache is not None and cache_kind == "cross" and "k" in cache:
            K = cache["k"]
            V = cache["v"]
            attn_mask = mask
        elif cache is not None and cache_kind == "self":
            K_new = self._split_heads(self.W_k(key))
            V_new = self._split_heads(self.W_v(value))
            if "k" in cache:
                K = torch.cat([cache["k"], K_new], dim=2)
                V = torch.cat([cache["v"], V_new], dim=2)
            else:
                K, V = K_new, V_new
            cache["k"] = K
            cache["v"] = V
            # Causality is automatic when stepping one token at a time.
            attn_mask = None
        else:
            K = self._split_heads(self.W_k(key))
            V = self._split_heads(self.W_v(value))
            if cache is not None and cache_kind == "cross":
                cache["k"] = K
                cache["v"] = V
            attn_mask = mask

        out, attn_w = scaled_dot_product_attention(Q, K, V, mask=attn_mask, scale=self.use_scale)
        out = self.dropout(out)
        self.last_attn_weights = attn_w.detach()

        out = self._merge_heads(out)
        return self.W_o(out)


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding (sec 3.5)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [L,1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, L, D]
        # Buffer (not a trainable parameter), as required.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embedding (used in the 2.4 ablation)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # [1, L]
        x = x + self.embedding(positions)
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, x W1 + b1) W2 + b2 (sec 3.3)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER LAYERS  (Post-LayerNorm, as in the original paper)
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Self-attention sub-layer
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        # Feed-forward sub-layer
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        layer_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        # Masked self-attention
        sa = self.self_attn(
            x, x, x, tgt_mask,
            cache=None if layer_cache is None else layer_cache.setdefault("self", {}),
            cache_kind="self" if layer_cache is not None else "none",
        )
        x = self.norm1(x + self.dropout1(sa))
        # Cross-attention over encoder memory
        ca = self.cross_attn(
            x, memory, memory, src_mask,
            cache=None if layer_cache is None else layer_cache.setdefault("cross", {}),
            cache_kind="cross" if layer_cache is not None else "none",
        )
        x = self.norm2(x + self.dropout2(ca))
        # Feed-forward
        ff = self.ffn(x)
        x = self.norm3(x + self.dropout3(ff))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

def _clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList(copy.deepcopy(module) for _ in range(n))


class Encoder(nn.Module):
    """Stack of N EncoderLayers + final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = _clones(layer, N)
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N DecoderLayers + final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = _clones(layer, N)
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: Optional[list] = None,
    ) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache[i]
            x = layer(x, memory, src_mask, tgt_mask, layer_cache=layer_cache)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════


# ─── helpers used by Transformer.__init__ for autograder bootstrap ──

class _SimpleVocab:
    """Tiny picklable Vocab; mirrors the API of dataset.Vocab."""

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def lookup_token(self, idx):
        return self.itos[idx]

    @classmethod
    def from_dict(cls, payload):
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, dict) and "itos" in payload:
            return cls(payload["itos"])
        if isinstance(payload, (list, tuple)):
            return cls(list(payload))
        raise ValueError("Unsupported vocab payload")

    def to_dict(self):
        return {"itos": list(self.itos)}


def _download_and_load_checkpoint(path: str, gdrive_id: Optional[str]) -> dict:
    """Fetch the checkpoint from Drive if missing, then torch.load it."""
    if not os.path.exists(path):
        if not gdrive_id or gdrive_id == "REPLACE_WITH_YOUR_GDRIVE_FILE_ID":
            raise FileNotFoundError(
                f"No checkpoint at {path!r} and GDRIVE_CHECKPOINT_ID is not set. "
                "Edit model.py and set GDRIVE_CHECKPOINT_ID to the Drive file ID "
                "of your trained .pt before submitting."
            )
        if gdown is None:
            raise RuntimeError(
                "gdown is required to download the checkpoint. "
                "Add `gdown` to requirements.txt."
            )
        gdown.download(id=gdrive_id, output=path, quiet=False)
    return torch.load(path, map_location="cpu")


def _load_spacy_tokenizer(model_name: str):
    """Lazy-load a spaCy tokenizer; returns a callable str -> spaCy Doc."""
    try:
        import spacy

        return spacy.load(model_name, disable=["tagger", "parser", "ner", "lemmatizer"])
    except Exception:
        # Fallback to whitespace if the spaCy model is not installed.
        class _Tok:
            def __call__(self, text):
                class _T:
                    def __init__(self, t): self.text = t
                return [_T(t) for t in text.split()]

        return _Tok()


def _detokenize_en(tokens: list) -> str:
    """Re-glue spaCy-style tokens into a natural English string.

    Handles standard punctuation that spaCy splits off (commas, periods,
    quotes, brackets, contractions). Improves BLEU vs naive ' '.join.
    """
    if not tokens:
        return ""
    out = ""
    no_space_before = {".", ",", ";", ":", "!", "?", ")", "]", "}", "'s",
                       "n't", "'re", "'ve", "'ll", "'d", "'m", "%", "..."}
    no_space_after = {"(", "[", "{", "$", "#", "@"}
    prev = None
    for tok in tokens:
        if prev is None:
            out = tok
        elif tok in no_space_before or prev in no_space_after:
            out += tok
        elif tok.startswith("'") and tok != "'":
            out += tok
        else:
            out += " " + tok
        prev = tok
    # Capitalise first letter for a cleaner output.
    return out[0].upper() + out[1:] if out else out


class Transformer(nn.Module):
    """Encoder-Decoder Transformer for seq2seq.

    Constructor accepts zero arguments so the autograder can call
    `Transformer()` and then `model.infer(...)`. When `src_vocab_size` /
    `tgt_vocab_size` are not given, the model self-bootstraps from a
    Google Drive checkpoint (see `GDRIVE_CHECKPOINT_ID` above) which
    must contain `model_config`, `model_state_dict`, and the source /
    target vocabularies + the source tokenizer name.
    """

    def __init__(
        self,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        d_model: int = 512,
        N: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        pad_idx: int = 1,
        positional_encoding: str = "sinusoidal",   # 'sinusoidal' | 'learned'
        checkpoint_path: Optional[str] = None,
        gdrive_id: Optional[str] = None,
    ) -> None:
        super().__init__()

        # ── Auto-bootstrap path: no architecture given ────────────────
        # Download the pre-trained checkpoint and read architecture +
        # vocab out of it before we build any submodule.
        bootstrap_state: Optional[dict] = None
        if src_vocab_size is None or tgt_vocab_size is None:
            ckpt_path = checkpoint_path or DEFAULT_CHECKPOINT_PATH
            gid = gdrive_id or GDRIVE_CHECKPOINT_ID
            bootstrap_state = _download_and_load_checkpoint(ckpt_path, gid)
            cfg = bootstrap_state.get("model_config", {})
            src_vocab_size = cfg.get("src_vocab_size", src_vocab_size)
            tgt_vocab_size = cfg.get("tgt_vocab_size", tgt_vocab_size)
            d_model = cfg.get("d_model", d_model)
            N = cfg.get("N", N)
            num_heads = cfg.get("num_heads", num_heads)
            d_ff = cfg.get("d_ff", d_ff)
            dropout = cfg.get("dropout", dropout)
            pad_idx = cfg.get("pad_idx", pad_idx)
            positional_encoding = cfg.get("positional_encoding", positional_encoding)

        if src_vocab_size is None or tgt_vocab_size is None:
            raise RuntimeError(
                "Transformer requires src/tgt vocab sizes either via constructor "
                "args or via a checkpoint that contains model_config."
            )

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.pad_idx = pad_idx
        self.positional_encoding = positional_encoding

        # Embeddings (scaled by sqrt(d_model) per sec 3.4)
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        if positional_encoding == "sinusoidal":
            self.src_pe = PositionalEncoding(d_model, dropout, max_len)
            self.tgt_pe = PositionalEncoding(d_model, dropout, max_len)
        elif positional_encoding == "learned":
            self.src_pe = LearnedPositionalEncoding(d_model, dropout, max_len)
            self.tgt_pe = LearnedPositionalEncoding(d_model, dropout, max_len)
        else:
            raise ValueError(f"Unknown positional_encoding: {positional_encoding}")

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        # Output projection to target vocab
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._init_parameters()

        # Stash for the inference helper.
        self._src_vocab = None
        self._tgt_vocab = None
        self._src_tokenizer = None

        # ── Load weights + vocab if a checkpoint is available ────────
        state = bootstrap_state
        if state is None and checkpoint_path is not None and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location="cpu")

        if state is not None:
            if isinstance(state, dict) and "model_state_dict" in state:
                self.load_state_dict(state["model_state_dict"], strict=False)
                if "src_vocab" in state and "tgt_vocab" in state:
                    self._src_vocab = _SimpleVocab.from_dict(state["src_vocab"])
                    self._tgt_vocab = _SimpleVocab.from_dict(state["tgt_vocab"])
                # Eagerly load tokenizer so the first infer() call isn't slow.
                self._src_tokenizer = _load_spacy_tokenizer(
                    state.get("src_spacy_model", "de_core_news_sm")
                )
            else:
                self.load_state_dict(state, strict=False)

        # Set inference mode by default to skip dropout in infer().
        self.eval()

    def _init_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _embed_src(self, src: torch.Tensor) -> torch.Tensor:
        return self.src_pe(self.src_embed(src) * math.sqrt(self.d_model))

    def _embed_tgt(self, tgt: torch.Tensor) -> torch.Tensor:
        return self.tgt_pe(self.tgt_embed(tgt) * math.sqrt(self.d_model))

    # ── AUTOGRADER HOOKS ─────────────────────────────────────────────

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self._embed_src(src), src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        dec_out = self.decoder(self._embed_tgt(tgt), memory, src_mask, tgt_mask)
        return self.generator(dec_out)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    # ── Inference helper ─────────────────────────────────────────────

    def attach_vocab(self, src_vocab, tgt_vocab, src_tokenizer):
        """Attach vocab + tokenizer so `infer` works end-to-end."""
        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab
        self._src_tokenizer = src_tokenizer

    def infer(self, src_sentence: str, max_len: int = 64, beam_size: int = 4,
              length_penalty: float = 0.6) -> str:
        """Translate a German sentence with KV-cached beam search.

        Args:
            src_sentence  : raw German text.
            max_len       : maximum generated length (in tokens).
            beam_size     : 1 = greedy, 4 = a good speed/quality tradeoff.
            length_penalty: GNMT-style alpha; higher favours longer outputs.

        Returns:
            Detokenized English string.
        """
        if self._src_vocab is None or self._tgt_vocab is None:
            raise RuntimeError(
                "No vocab attached. Either construct Transformer() with a "
                "checkpoint that embeds vocabularies, or call attach_vocab()."
            )
        if self._src_tokenizer is None:
            self._src_tokenizer = _load_spacy_tokenizer("de_core_news_sm")

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()

        sos = self._tgt_vocab.stoi["<sos>"]
        eos = self._tgt_vocab.stoi["<eos>"]
        unk = self._src_vocab.stoi.get("<unk>", 0)

        tokens = (
            ["<sos>"]
            + [t.text.lower() for t in self._src_tokenizer(src_sentence)]
            + ["<eos>"]
        )
        src_ids = [self._src_vocab.stoi.get(t, unk) for t in tokens]
        src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
        src_mask = make_src_mask(src, self.pad_idx)

        with torch.no_grad():
            memory = self.encode(src, src_mask)              # [1, S, D]
            if beam_size == 1:
                out_ids = self._greedy_with_cache(memory, src_mask, sos, eos, max_len)
            else:
                out_ids = self._beam_search(
                    memory, src_mask, sos, eos, max_len, beam_size, length_penalty,
                )

        if was_training:
            self.train()

        return _detokenize_en([self._tgt_vocab.itos[i] for i in out_ids])

    # ── private decoders ─────────────────────────────────────────────

    def _step_embed(self, ys: torch.Tensor) -> torch.Tensor:
        """Embed only the most recent column of `ys`, with the right PE."""
        tgt_in = ys[:, -1:]                                  # [B, 1]
        step_idx = ys.size(1) - 1
        emb = self.tgt_embed(tgt_in) * math.sqrt(self.d_model)
        if isinstance(self.tgt_pe, PositionalEncoding):
            emb = emb + self.tgt_pe.pe[:, step_idx : step_idx + 1, :]
        else:
            pos = torch.full((ys.size(0), 1), step_idx, device=ys.device, dtype=torch.long)
            emb = emb + self.tgt_pe.embedding(pos)
        return emb

    def _greedy_with_cache(self, memory, src_mask, sos, eos, max_len):
        device = memory.device
        cache = [{} for _ in range(self.N)]
        ys = torch.tensor([[sos]], dtype=torch.long, device=device)
        out_ids = []
        for _ in range(max_len):
            emb = self._step_embed(ys)
            dec_out = self.decoder(emb, memory, src_mask, tgt_mask=None, cache=cache)
            next_token = int(self.generator(dec_out[:, -1]).argmax(dim=-1).item())
            if next_token == eos:
                break
            out_ids.append(next_token)
            ys = torch.cat([ys, torch.tensor([[next_token]], device=device, dtype=torch.long)], dim=1)
        return out_ids

    def _beam_search(self, memory, src_mask, sos, eos, max_len, beam_size, alpha):
        """KV-cached beam search with length-normalised re-ranking."""
        device = memory.device
        V = self.tgt_vocab_size

        # Expand encoder outputs across beams.
        memory = memory.expand(beam_size, -1, -1).contiguous()
        src_mask = src_mask.expand(beam_size, -1, -1, -1).contiguous()

        ys = torch.full((beam_size, 1), sos, dtype=torch.long, device=device)
        # Only beam 0 is "real" at the first step so we don't pick `beam_size`
        # identical extensions of <sos>.
        scores = torch.full((beam_size,), -1e9, device=device)
        scores[0] = 0.0
        cache = [{} for _ in range(self.N)]

        finished_seqs: list[list[int]] = []
        finished_scores: list[float] = []

        def lp(length):
            return ((5 + length) / 6.0) ** alpha

        for _ in range(max_len):
            emb = self._step_embed(ys)                      # [beam, 1, D]
            dec_out = self.decoder(emb, memory, src_mask, tgt_mask=None, cache=cache)
            logits = self.generator(dec_out[:, -1])          # [beam, V]
            log_probs = torch.log_softmax(logits, dim=-1)
            cumulative = scores.unsqueeze(1) + log_probs     # [beam, V]
            flat = cumulative.view(-1)                       # [beam * V]

            top_scores, top_idx = flat.topk(beam_size)       # both [beam]
            parent = top_idx // V
            token = top_idx % V

            ys = torch.cat([ys.index_select(0, parent), token.unsqueeze(1)], dim=1)

            # Reorder KV caches by parent.
            for layer_cache in cache:
                for kind in ("self", "cross"):
                    sub = layer_cache.get(kind)
                    if sub is None:
                        continue
                    if "k" in sub:
                        sub["k"] = sub["k"].index_select(0, parent)
                        sub["v"] = sub["v"].index_select(0, parent)

            scores = top_scores.clone()

            # Pull off any beam that just emitted <eos>, dock its slot so it
            # can't grow further.
            for i in range(beam_size):
                if int(token[i].item()) == eos:
                    seq = ys[i, 1:-1].tolist()  # drop <sos> and the final <eos>
                    finished_seqs.append(seq)
                    finished_scores.append(top_scores[i].item() / lp(len(seq) + 1))
                    scores[i] = -1e9

            if len(finished_seqs) >= beam_size:
                break

        # If nothing finished within max_len, fall back to the live beams.
        if not finished_seqs:
            for i in range(beam_size):
                seq = ys[i, 1:].tolist()
                finished_seqs.append(seq)
                finished_scores.append(scores[i].item() / lp(len(seq) + 1))

        best = max(range(len(finished_seqs)), key=lambda i: finished_scores[i])
        return finished_seqs[best]
