from torch import nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import json


class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, n_embd: int, head_size: int, dropout: float):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # X is of size (B, T, n_embd)
        # Mask of size (T, T)
        B, T, C = x.shape
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)  # (B, T, head_size)

        wei = q @ k.transpose(1, 2)  # (B, T, Head_size) @ (B, Head_size, T) = (B, T, T)
        # Normalize the weights
        wei = wei * C**-0.5

        # Mask the upper triangular part of the matrix
        if mask is not None:
            wei = wei.masked_fill(
                mask == 0, float("-inf")
            )  # We mask the upper triangle. We set -inf to the upper triangle so after softmax it will be zero

        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # Perform the weighted sum
        v = self.value(x)  # (B, T, head_size)

        out = wei @ v  # (B, T, T) @# (B, T, head_size) = (B, T, head_size)

        return out


class CrossAttentionHead(nn.Module):
    def __init__(self, n_embd: int, head_size: int, dropout: float):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output):
        # X is of size (B, T, n_embd)
        # enc_output of size (B, T, n_embd)
        B, T, C = x.shape
        q = self.query(x)  # (B, T, head_size)
        k = self.key(enc_output)  # (B, T, head_size)

        wei = q @ k.transpose(1, 2)  # (B, T, Head_size) @ (B, Head_size, T) = (B, T, T)
        # Normalize the weights
        wei = wei * C**-0.5

        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # Perform the weighted sum
        v = self.value(enc_output)  # (B, T, head_size)

        out = wei @ v  # (B, T, T) @# (B, T, head_size) = (B, T, head_size)

        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads: int, head_size: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, dropout) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        head_results = [h(x, mask) for h in self.heads]  # Num heads x (B, T, head_size)
        head_results = torch.cat(head_results, dim=-1)  # (B, T, num_heads * head_size)
        out = self.linear(head_results)  # (B, T, n_embd)
        out = self.dropout(out)
        return out


class MultiHeadCrossAttention(nn.Module):
    """multiple heads of cross-attention in paralell"""

    def __init__(self, num_heads: int, head_size: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList(
            [CrossAttentionHead(n_embd, head_size, dropout) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output):
        head_results = [
            h(x, enc_output) for h in self.heads
        ]  # Num heads x (B, T, head_size)
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


class EncoderBlock(nn.Module):
    def __init__(self, n_embd: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, n_embd // num_heads, dropout)
        self.feed_forward = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):  # (B, T, n_embd)
        # x + * is the residual connection
        x = x + self.attention(self.ln1(x), mask=None)  # (B, T, n_embd)
        x = x + self.feed_forward(self.ln2(x))  # (B, T, n_embd)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, n_embd: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, n_embd // num_heads, dropout)
        self.cross_attention = MultiHeadCrossAttention(
            num_heads, n_embd // num_heads, dropout
        )
        self.feed_forward = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, x, enc_output, mask):  # (B, T, n_embd), (B, T, n_embd)
        # x + * is the residual connection
        x = x + self.attention(self.ln1(x), mask=mask)  # (B, T, n_embd)
        x = x + self.cross_attention(self.ln2(x), enc_output)  # (B, T, n_embd)
        x = x + self.feed_forward(self.ln3(x))  # (B, T, n_embd)
        return x


class T5(nn.Module):
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
        self.encoder = nn.Sequential(
            *[EncoderBlock(n_embd, num_heads, dropout) for _ in range(num_blocks)]
            + [nn.LayerNorm(n_embd)]
        )

        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(n_embd, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.decoder_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.register_buffer(
            "tril", torch.tril(torch.ones(max_seq_length, max_seq_length)).to(device)
        )

    def forward(self, encoder_tokens, decoder_tokens, targets=None) -> torch.Tensor:
        B, T = encoder_tokens.shape
        encoder_embeddings = self.token_embedding_table(
            encoder_tokens
        )  # (B, T, n_embd)
        encoder_pos_embeddings = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, n_embd)
        encoder_embeddings = (
            encoder_embeddings + encoder_pos_embeddings
        )  # (B, T, n_embd)
        enc_output = self.encoder(encoder_embeddings)  # (B, T, n_embd)

        B, T = decoder_tokens.shape
        decoder_embeddings = self.token_embedding_table(
            decoder_tokens
        )  # (B, T, n_embd)
        decoder_pos_embeddings = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, n_embd)

        decoder_embeddings = (
            decoder_embeddings + decoder_pos_embeddings
        )  # (B, T, n_embd)
        mask = self.tril[:T, :T]  # (T, T)
        dec_output = decoder_embeddings
        for block in self.decoder_blocks:
            dec_output = block(dec_output, enc_output, mask)  # (B, T, n_embd)

        dec_output = self.decoder_ln(dec_output)  # (B, T, n_embd)
        logits = self.lm_head(dec_output)  # (B, T, vocab_size)

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
        B, T = idx.shape
        decoder_tokens = (
            torch.tensor([bos_token_idx] * B).unsqueeze(1).to(device)
        )  # (B, 1)
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, _ = self(
                encoder_tokens=idx, decoder_tokens=decoder_tokens, targets=None
            )
            # Focus only on the last time step
            logits = logits[:, -1, :]  # Becomes (Batch Size, Vocab Size)
            # Forbit bos token
            logits[:, bos_token_idx] = float("-inf")
            # Apply softmax to tget probabilities
            probs = F.softmax(logits, dim=-1)  # (Batch Size, Vocab Size)
            # Get token index with highest probability
            idx_next = torch.multinomial(probs, num_samples=1)  # (Batch Size, 1)
            # Apprend samples index tot he running sequence
            decoder_tokens = torch.cat(
                (decoder_tokens, idx_next), dim=1
            )  # (Batch Size, Curent Sequence_length +1)

        return decoder_tokens


@torch.no_grad()
def evaluate_model():
    out = []
    model.eval()
    with tqdm(total=eval_iters, desc="Evaluation", leave=False) as pbar:
        for _ in range(eval_iters):
            X, Y = get_batch("val")
            B, T = Y.shape
            _, loss = model(
                encoder_tokens=X.to(device),
                decoder_tokens=Y.to(device),
                targets=Y.to(device),
            )
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
            _, loss = model(xb.to(device), yb.to(device), yb.to(device))
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
    ix = torch.randint(len(data), (batch_size,))

    x = [torch.tensor(data[i]["en"]) for i in ix]
    y = [torch.tensor([bos_token_idx] + data[i]["es"] + [eos_token_idx]) for i in ix]

    x = torch.nn.utils.rnn.pad_sequence(
        x, batch_first=True, padding_value=pad_token_idx
    )
    y = torch.nn.utils.rnn.pad_sequence(
        y, batch_first=True, padding_value=pad_token_idx
    )

    return x, y


with open("data/translation.json", "r", encoding="utf8") as f:
    data = json.load(f)

data = [{"en": d["en"].lower(), "es": d["es"].lower()} for d in data]


# Hparams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
batch_size = 4  # how many independent sequence will be process in parallel?
block_size = 256  # what is the maximun context length for predictions?
n_embd = 128  # Embedding size
num_heads = 4  # How many heads in the multi-head attention
num_blocks = 4  # How many blocks in the model
dropout = 0.2  # Dropout rate
num_iterations = 10000  # how many iterations to train for
eval_iters = 100  # how many iters between evaluations
learning_rate = 3e-4

print("=== Build the tokenizer ===")
text = "".join([d["en"] for d in data] + [d["es"] for d in data])
bos_token = "#"
pad_token = "_"
eos_token = "!"

chars = sorted(list(set(text)) + [bos_token, pad_token, eos_token])
vocab_size = len(chars)
print("Vocabulary:")
print("".join(chars))
print(f"Vocabulary size: {vocab_size}")
print()

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

bos_token_idx = stoi[bos_token]
pad_token_idx = stoi[pad_token]
eos_token_idx = stoi[eos_token]
print(f"BOS token index: {bos_token_idx}")

print("=== Load data ===")
data = [{"en": encode(d["en"]), "es": encode(d["es"])} for d in data]


n = int(0.9 * len(data))
train_data = data  # [:n]
val_data = data  # [n:]

print(f"Training data lengh: {len(train_data)}")
print(f"Val data length: {len(val_data)}")


model = T5(
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
X, Y = get_batch("val")
for input_sentence, expected_output in zip(X, Y):
    input_sentence = "".join([itos[i.item()] for i in input_sentence])
    print(input_sentence)
    expected_output = "".join([itos[i.item()] for i in expected_output])
    print(expected_output)
    print()


model_outputs = model.generate(X.to(device), 128)
for sent in model_outputs:
    sent = "".join([itos[i.item()] for i in sent])
    print(sent)
