import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import requests


# -----------------------------
# Model Components Definitions
# -----------------------------

class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """Feed-forward neural network."""
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """Transformer block with attention and MLP."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = None  # Will set this later based on tokenizer
    n_layer: int = 6        # Reduced layers for faster training
    n_head: int = 6         # Reduced heads
    n_embd: int = 384       # Reduced embedding size
    dropout: float = 0.2
    bias: bool = True       # Use bias in Linear and LayerNorm layers

class StartLayer(nn.Module):
    """Initial layer that applies token and positional embeddings and dropout."""
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        pos = pos.unsqueeze(0).expand(b, t)  # shape (b, t)
        tok_emb = self.wte(idx)  # token embeddings
        pos_emb = self.wpe(pos)  # position embeddings
        x = self.drop(tok_emb + pos_emb)
        return x

class LNFLayer(nn.Module):
    """Final LayerNorm layer."""
    def __init__(self, config):
        super().__init__()
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.ln_f(x)
        return x

class LMHeadLayer(nn.Module):
    """Language Modeling Head that projects embeddings to vocabulary size."""
    def __init__(self, config):
        super().__init__()
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x):
        logits = self.lm_head(x)
        return logits

# -----------------------------
# Model Dictionary Definition
# -----------------------------

def get_model_dict(config):
    model = {}

    # Start layer (embedding and positional encoding)
    model['start'] = {
        'callable': {'object': StartLayer, 'settings': {'config': config}},
        'dst': {'to': ['block_0']},
        'rcv': {'src': [], 'strategy': None},
        'stage': 1,
        'num_layer_shards': 1,
    }

    # Transformer blocks
    for i in range(config.n_layer):
        model[f'block_{i}'] = {
            'callable': {'object': Block, 'settings': {'config': config}},
            'dst': {'to': [f'block_{i+1}'] if i+1 < config.n_layer else ['ln_f']},
            'rcv': {'src': [f'block_{i-1}'] if i > 0 else ['start'], 'strategy': None},
            'stage': 1,
            'num_layer_shards': 1,
        }

    # Final LayerNorm layer
    model['ln_f'] = {
        'callable': {'object': LNFLayer, 'settings': {'config': config}},
        'dst': {'to': ['lm_head']},
        'rcv': {'src': [f'block_{config.n_layer - 1}'], 'strategy': None},
        'stage': 1,
        'num_layer_shards': 1,
    }

    # Language Modeling Head
    model['lm_head'] = {
        'callable': {'object': LMHeadLayer, 'settings': {'config': config}},
        'dst': {'to': []},
        'rcv': {'src': ['ln_f'], 'strategy': None},
        'stage': 1,
        'num_layer_shards': 1,
    }

    return model

# -----------------------------
# Build the Model from Dictionary
# -----------------------------

class GPTModelFromDict(nn.Module):
    def __init__(self, model_dict):
        super().__init__()
        self.model_dict = model_dict
        self.layers = nn.ModuleDict()
        self.build_model()

        # Weight tying between token embeddings and lm_head
        self.layers['lm_head'].lm_head.weight = self.layers['start'].wte.weight

    def build_model(self):
        # Instantiate layers
        for name, layer_info in self.model_dict.items():
            callable_obj = layer_info['callable']['object']
            settings = layer_info['callable']['settings']
            self.layers[name] = callable_obj(**settings)

    def forward(self, idx, targets=None):
        layer_outputs = {}
        # Forward pass through layers in order
        for name in self.layers:
            layer = self.layers[name]
            # Get inputs from sources
            src_names = self.model_dict[name]['rcv']['src']
            if src_names:
                inputs = [layer_outputs[src] for src in src_names]
                # Apply strategy if any
                strategy = self.model_dict[name]['rcv']['strategy']
                if strategy:
                    x = strategy(*inputs)
                else:
                    x = inputs[0]  # Assuming single input if no strategy
            else:
                # For 'start' layer, input is idx
                x = idx
            # Forward through the layer
            x = layer(x)
            layer_outputs[name] = x

        logits = layer_outputs['lm_head']
        if targets is not None:
            # Compute loss
            B, T, C = logits.size()
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            return logits, None

# -----------------------------
# Prepare the Shakespeare Dataset
# -----------------------------

def download_shakespeare():
    url = 'https://gist.githubusercontent.com/blakesanie/dde3a2b7e698f52f389532b4b52bc254/raw/76fe1b5e9efcf0d2afdfd78b0bfaa737ad0a67d3/shakespeare.txt'
    url = 'https://github.com/jcjohnson/torch-rnn/blob/master/data/tiny-shakespeare.txt'  # tiny dataset for testing
    response = requests.get(url)
    data = response.text
    return data

# Simple character-level tokenizer
class CharTokenizer:
    def __init__(self, data):
        self.chars = sorted(list(set(data)))
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }

    def encode(self, s):
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, l):
        return ''.join([self.itos.get(i, '') for i in l])

# Custom dataset class
class ShakespeareDataset(Dataset):
    def __init__(self, data, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = data
        self.tokenized_data = tokenizer.encode(data)

    def __len__(self):
        return len(self.tokenized_data) - self.block_size

    def __getitem__(self, idx):
        x = self.tokenized_data[idx:idx+self.block_size]
        y = self.tokenized_data[idx+1:idx+self.block_size+1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# -----------------------------
# Training Loop Setup
# -----------------------------

# Download and prepare data
print("Downloading Shakespeare dataset...")
data = download_shakespeare()
print("Dataset downloaded.")

tokenizer = CharTokenizer(data)
config = GPTConfig(
    block_size=256,
    vocab_size=tokenizer.vocab_size,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2,
    bias=True
)

train_dataset = ShakespeareDataset(data, tokenizer, config.block_size)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Get the model dictionary and instantiate the model
model_dict = get_model_dict(config)
model = GPTModelFromDict(model_dict)

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Using device: {device}")

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
epochs = 100
model.train()
print("Starting training...")
for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (batch_idx+1)
    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {avg_loss:.4f}")
    total_loss = 0
print("Training completed.")

# -----------------------------
# Generate Text
# -----------------------------

def generate(model, idx, max_new_tokens, tokenizer):
    model.eval()
    idx = idx.to(device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -config.block_size:]  # Crop context if needed
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # Focus on the last time step
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    return idx

# Generate text
start_prompt = "To be, or not to be:"
start_tokens = tokenizer.encode(start_prompt)
start_tokens = torch.tensor(start_tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension
generated_tokens = generate(model, start_tokens, max_new_tokens=200, tokenizer=tokenizer)
generated_text = tokenizer.decode(generated_tokens[0].tolist())
print("\nGenerated Text:\n")
print(generated_text)

# save the model params
torch.save(model.state_dict(), 'shakespeare_model.pth')
