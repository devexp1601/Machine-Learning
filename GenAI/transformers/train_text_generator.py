import argparse
from pathlib import Path
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

"""Train a small decoder-only transformer for character-level text generation.

This script reads plain text, learns to predict the next character, and then
generates new text from the trained model.

Quick run (from this folder):
    python train_text_generator.py --data-path ../../data.txt

See --help for all hyperparameters.
"""


class Head(nn.Module):
    """Single causal self-attention head."""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Causal mask so each token only attends to current and previous tokens.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t, _ = x.shape
        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
        weights = weights.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        v = self.value(x)
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    """Run multiple attention heads in parallel, then mix their outputs."""

    def __init__(self, n_head: int, n_embd: int, block_size: int, dropout: float) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """Position-wise MLP used after attention in each transformer block."""

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block: LayerNorm + self-attention + feed-forward."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_head, n_embd, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerLanguageModel(nn.Module):
    """Decoder-only language model that predicts the next token."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, t = idx.shape

        # Combine token identity and token position information.
        token_embeddings = self.token_embedding_table(idx)
        positions = torch.arange(t, device=idx.device)
        position_embeddings = self.position_embedding_table(positions)
        x = token_embeddings + position_embeddings

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.view(batch_size * t, -1)
            targets_flat = targets.view(batch_size * t)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # Repeatedly predict one token at a time and append it to the sequence.
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def parse_args() -> argparse.Namespace:
    """Read command-line hyperparameters."""

    example_text = (
        "Examples:\n"
        "  python train_text_generator.py --data-path ../../data.txt\n"
        "  python train_text_generator.py --max-iters 5000 --save-path checkpoints/model.pt\n"
        "  python train_text_generator.py --batch-size 32 --block-size 64"
    )
    parser = argparse.ArgumentParser(
        description="Train a text-generation transformer with PyTorch",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=example_text,
    )
    parser.add_argument("--data-path", type=str, default="../../data.txt", help="Path to training text file")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-embd", type=int, default=192)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--generate-tokens", type=int, default=300)
    parser.add_argument("--save-path", type=str, default="model.pt", help="Path to save model checkpoint")
    return parser.parse_args()


def make_batch_fn(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
) -> Callable[[str], Tuple[torch.Tensor, torch.Tensor]]:
    """Create a function that returns random (x, y) training batches."""

    def get_batch(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data = train_data if split == "train" else val_data
        idx = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in idx])
        # Targets are inputs shifted by one token (next-token prediction).
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
        return x.to(device), y.to(device)

    return get_batch


@torch.no_grad()
def estimate_loss(
    model: TransformerLanguageModel,
    get_batch: Callable[[str], Tuple[torch.Tensor, torch.Tensor]],
    eval_iters: int,
) -> Dict[str, float]:
    """Average loss over several mini-batches for train/validation splits."""

    losses: Dict[str, float] = {}
    model.eval()
    for split in ("train", "val"):
        split_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            split_losses[k] = loss.item()
        losses[split] = split_losses.mean().item()
    model.train()
    return losses


def main() -> None:
    """Load data, train the model, save checkpoint, and print a sample."""

    args = parse_args()

    # Reproducibility and hardware selection.
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training text not found: {data_path.resolve()}")

    text = data_path.read_text(encoding="utf-8")
    if len(text) < args.block_size + 2:
        raise ValueError("Text is too short for the selected block size")

    # Build character-level vocabulary and lookup tables.
    vocab = sorted(set(text))
    vocab_size = len(vocab)

    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}

    def encode(s: str) -> list[int]:
        return [stoi[ch] for ch in s]

    def decode(tokens: list[int]) -> str:
        return "".join(itos[i] for i in tokens)

    # Encode full text and split into train/validation partitions.
    data = torch.tensor(encode(text), dtype=torch.long)
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    get_batch = make_batch_fn(
        train_data=train_data,
        val_data=val_data,
        batch_size=args.batch_size,
        block_size=args.block_size,
        device=device,
    )

    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Main optimization loop.
    for step in range(args.max_iters + 1):
        if step % args.eval_interval == 0:
            stats = estimate_loss(model, get_batch, args.eval_iters)
            print(
                f"step {step:5d} | train loss: {stats['train']:.4f} | val loss: {stats['val']:.4f}"
            )

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save weights + tokenizer tables so this model can be reused later.
    model_path = Path(args.save_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
            "config": {
                "block_size": args.block_size,
                "n_embd": args.n_embd,
                "n_head": args.n_head,
                "n_layer": args.n_layer,
                "dropout": args.dropout,
                "vocab_size": vocab_size,
            },
        },
        model_path,
    )
    print(f"Model saved to {model_path.resolve()}")

    # Quick qualitative check: generate sample text from the trained model.
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=args.generate_tokens)[0].tolist()
    print("\nSample generated text:\n")
    print(decode(generated))


if __name__ == "__main__":
    main()
