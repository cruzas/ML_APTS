import os
import torch
import torch.nn as nn
import math
from transformers import GPT2TokenizerFast
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader

print("Entered test_wikipedia.py...")


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Define linear layers for query, key, and value
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        # Output linear layer
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # query, key, value shape: (batch_size, seq_len, d_model)
        batch_size = query.size(0)

        # Linear projections and split into heads
        Q = self.linear_q(query).view(
            batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.linear_k(key).view(
            batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.linear_v(value).view(
            batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # Expand mask to match the dimensions of scores
            # Shape: (1, 1, seq_len, seq_len)
            mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)

        # Concatenate heads
        context = torch.matmul(attn, V).transpose(
            1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.linear_out(context)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len):
        super(LanguageModel, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.token_embedding(x)
        x = self.position_embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        logits = self.fc_out(x)
        return logits


def generate_subsequent_mask(seq_len):
    # Generate a mask where 'True' indicates positions to be masked
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask  # Shape: (seq_len, seq_len)


def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight.data)


# Tokenizer and dataset setup
print("Creating tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
# Ensure there is a pad token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
vocab_size = len(tokenizer)

print("Loading dataset...")
train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1",
                             split="train", cache_dir="./tests/wikitext-2-raw-v1")
test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1",
                            split="test", cache_dir="./tests/wikitext-2-raw-v1")


def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)


print("Tokenizing dataset...")
tokenized_train_dataset = train_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"])
tokenized_train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask"])

tokenized_test_dataset = test_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"])
tokenized_test_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask"])

print("Creating dataloaders...")
train_dataloader = DataLoader(
    tokenized_train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(
    tokenized_test_dataset, batch_size=32, shuffle=False)

# Model parameters
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
max_seq_len = 128
num_epochs = 2
learning_rate = 1e-5

model = LanguageModel(vocab_size, d_model, num_heads,
                      d_ff, num_layers, max_seq_len)

# Check if SLM_model_wikitext-2-raw-v1.pth exists
if os.path.exists("SLM_model_wikitext-2-raw-v1.pth"):
    model.load_state_dict(torch.load("SLM_model_wikitext-2-raw-v1.pth"))
    model.to('cuda')
else:
    model.apply(initialize_weights)


TEST = True
if TEST:
    while True:
        def generate_text(model, tokenizer, prompt, max_length=50):
            model.eval()
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
            generated = input_ids

            with torch.no_grad():
                for _ in range(max_length):
                    seq_len = generated.size(1)
                    mask = generate_subsequent_mask(seq_len).to('cuda')
                    logits = model(generated, mask=mask)
                    next_token_logits = logits[:, -1, :]

                    # Optionally apply temperature or top-k sampling
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

                    generated = torch.cat((generated, next_token), dim=1)

                    # Stop generation if the model outputs the end-of-sequence token
                    if next_token.item() == tokenizer.eos_token_id:
                        break

            output_text = tokenizer.decode(generated.squeeze().tolist(), skip_special_tokens=True)
            return output_text


        # Prompt the user for a question
        user_question = input("Enter your question: ")

        # Generate an answer using the trained model
        answer = generate_text(model, tokenizer, user_question, max_length=50)

        print("\nModel's Answer:")
        print(answer)




criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model = model.to('cuda')
prev_loss = 0.15618231819897163  # float('inf')

print("Training model...")
with torch.autograd.detect_anomaly():
    for epoch in range(num_epochs):
        avg_loss = 0
        for i, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to('cuda')

            optimizer.zero_grad()

            seq_len = input_ids.size(1)
            mask = generate_subsequent_mask(seq_len).to(input_ids.device)

            logits = model(input_ids, mask=mask)
            logits = logits.view(-1, vocab_size)
            target_ids = input_ids.view(-1)

            loss = criterion(logits, target_ids)
            loss.backward()
            avg_loss += loss.item()

            # Clip gradients to avoid exploding gradient issue
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        avg_loss /= len(train_dataloader)
        if prev_loss > avg_loss:
            torch.save(model.state_dict(), f"SLM_model_wikitext-2-raw-v1.pth")

        prev_loss = avg_loss
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")

        # do validation
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to('cuda')

                seq_len = input_ids.size(1)
                mask = generate_subsequent_mask(seq_len).to(input_ids.device)

                logits = model(input_ids, mask=mask)
                logits = logits.view(-1, vocab_size)
                target_ids = input_ids.view(-1)

                loss = criterion(logits, target_ids)
                val_loss += loss.item()
            val_loss /= len(test_dataloader)
            print(f"Validation Loss: {val_loss}")

        model.train()



