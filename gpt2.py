from torch import nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


class Head(nn.Module):
    """one head of self attention"""

    def __init__(
        self, n_embd: int, head_size: int, max_seq_length: int, dropout: float
    ):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(max_seq_length, max_seq_length)).to(device)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # X is of size (B, T, n_embd)
        B, T, C = x.shape
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)  # (B, T, head_size)

        wei = q @ k.transpose(1, 2)  # (B, T, Head_size) @ (B, Head_size, T) = (B, T, T)
        # Normalize the weights
        wei = wei * C**-0.5

        # Mask the upper triangular part of the matrix
        tril = self.tril[:T, :T]  # Mask have size Time x Time.
        wei = wei.masked_fill(
            tril == 0, float("-inf")
        )  # We mask the upper triangle. We set -inf to the upper triangle so after softmax it will be zero

        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # Perform the weighted sum
        v = self.value(x)  # (B, T, head_size)

        out = wei @ v  # (B, T, T) @# (B, T, head_size) = (B, T, head_size)

        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in prallel"""

    def __init__(
        self, num_heads: int, head_size: int, max_seq_length: int, dropout: float
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, max_seq_length, dropout) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        head_results = [h(x) for h in self.heads]  # Num heads x (B, T, head_size)
        head_results = torch.cat(head_results, dim=-1)  # (B, T, num_heads * head_size)
        out = self.linear(head_results)  # (B, T, n_embd)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)  # Token level feed forward (B, T, n_embd) -> (B, T, n_embd)


class LayerNorm(nn.Module):
    """
    Layer normalization module.
    Runs at token level (Batch and time are batch dimensions)
    Each token is unit mean and unit gaussian
    """

    def __init__(self, n_embd: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = torch.ones(n_embd).to(device)
        self.beta = torch.zeros(n_embd).to(device)

    def __call__(self, x):
        xmean = x.mean(1, keepdim=True)  # (B, 1, n_embd)
        xvar = x.var(1, keepdim=True)  # (B, 1, n_embd)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # Normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Block(nn.Module):
    def __init__(
        self, n_embd: int, num_heads: int, max_seq_length: int, dropout: float
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            num_heads, n_embd // num_heads, max_seq_length, dropout
        )
        self.feed_forward = FeedForward(n_embd, dropout)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

    def forward(self, x):  # (B, T, n_embd)
        # x + * is the residual connection
        x = x + self.attention(self.ln1(x))  # (B, T, n_embd)
        x = x + self.feed_forward(self.ln2(x))  # (B, T, n_embd)
        return x


class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        max_seq_length: int,
        num_heads: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(max_seq_length, n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(n_embd, num_heads, max_seq_length, dropout)
                for _ in range(num_blocks)
            ]
            + [LayerNorm(n_embd)]
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None) -> torch.Tensor:
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_embeddings = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, n_embd)
        x = token_embeddings + pos_embeddings  # (B, T, n_embd)
        x = self.blocks(x)  # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens: int):
        # idx is (B, Current Sequence Length) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to max_len
            idx_cond = idx[:, -block_size:]
            # Get the predictions
            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # Becomes (Batch Size, Vocab Size)
            # Apply softmax to tget probabilities
            probs = F.softmax(logits, dim=-1)  # (Batch Size, Vocab Size)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (Batch Size , 1)
            # Apprend samples index tot he running sequence
            idx = torch.cat(
                (idx, idx_next), dim=1
            )  # (Batch Size, Curent Sequence_length +1)

            yield idx_next

@torch.no_grad()
def evaluate_model():
    out = []
    model.eval()
    with tqdm(total=eval_iters, desc="Evaluation", leave=False) as pbar:
        for _ in range(eval_iters):
            X, Y = get_batch("val")
            _, loss = model(X.to(device), Y.to(device))
            out.append(loss.item())
            pbar.update(1)
    model.train()
    return sum(out) / len(out)


def train_model(num_iterations):
    train_losses = []
    eval_losses = []

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    with tqdm(
        total=num_iterations, desc="Current Train Loss: nan. Current Eval Loss: nan"
    ) as pbar:
        current_eval_loss = float("inf")
        for step in range(num_iterations):
            xb, yb = get_batch("train")

            # Evalate the loss
            _, loss = model(xb.to(device), yb.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.update(1)

            if step % eval_iters == 0:
                eval_loss = evaluate_model()
                eval_losses.append(eval_loss)
                current_eval_loss = eval_loss

            pbar.set_description(
                f"Current Train Loss: {loss.item():.4f}. Current Eval Loss: {current_eval_loss:.4f}"
            )

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(train_losses)), train_losses)
    plt.xlabel("Training steps")
    plt.ylabel("Loss")
    plt.title(f"Training Loss: {train_losses[-1]:.4f}")
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(eval_losses)), eval_losses)
    plt.xlabel("Training steps")
    plt.ylabel("Loss")
    plt.title(f"Evaluation Loss: {eval_losses[-1]:.4f}")
    plt.show()


def encode(s):
    # Encoder: take a string, output a list of integers
    return [stoi[c] for c in s]


def decode(l):
    # Decoder: take a list of integers, output a string
    return "".join([itos[i] for i in l])


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


with open("data/NEWHOPE.TXT", "r", encoding="utf8") as f:
    text = f.read()


# Hparams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
batch_size = 64  # how many independent sequence will be process in parallel?
block_size = 256  # what is the maximun context length for predictions?
n_embd = 128  # Embedding size
num_heads = 4  # How many heads in the multi-head attention
num_blocks = 4  # How many blocks in the model
dropout = 0.2  # Dropout rate
num_iterations = 5000  # how many iterations to train for
eval_iters = 200  # how many iters between evaluations
learning_rate = 3e-4

print("=== Build the tokenizer ===")
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Vocabulary:")
print("".join(chars))
print(f"Vocabulary size: {vocab_size}")
print()

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

print("=== Load data ===")
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Dataset shape: {data.shape}")

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Training data lengh: {len(train_data)}")
print(f"Val data length: {len(val_data)}")


model = GPT2(
    vocab_size=vocab_size,
    n_embd=n_embd,
    max_seq_length=block_size,
    num_heads=num_heads,
    num_blocks=num_blocks,
    dropout=dropout,
).to(device)
print("Model:")
print(model)

print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

print("=== Training ===")
train_model(num_iterations)

print("=== Generation ===")
model.eval()
context = "Luke, I am your father. "
print(context, end="")
context = torch.tensor(encode(context), dtype=torch.long).unsqueeze(0).to(device)

for idx in model.generate(context, 1000):
    print(itos[idx[0].item()], end="")
