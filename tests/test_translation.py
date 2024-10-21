import torch
import torch.nn as nn
import math
from transformers import GPT2Tokenizer
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader

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
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
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
        
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        
        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
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

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len):
        super(TransformerEncoder, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
    
    def forward(self, x, mask=None):
        x = self.token_embedding(x)
        x = self.position_embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len):
        super(TransformerDecoder, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, enc_output, mask=None):
        x = self.token_embedding(x)
        x = self.position_embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc_out(x)

def generate_subsequent_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask  # Shape: (seq_len, seq_len)

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight.data)

# Tokenizer and dataset setup
src_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tgt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<pad>', eos_token='</s>')

vocab_size_src = len(src_tokenizer)
vocab_size_tgt = len(tgt_tokenizer)

dataset = load_dataset('wmt14', 'fr-en')

def tokenize_function(examples):
    tgt_texts = examples['translation']['en']
    src_texts = examples['translation']['fr']
    return {
        'src': src_tokenizer(src_texts, truncation=True, padding="max_length", max_length=128)['input_ids'],
        'tgt': tgt_tokenizer(tgt_texts, truncation=True, padding="max_length", max_length=128)['input_ids'],
    }

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["src", "tgt"])
dataloader = DataLoader(tokenized_dataset, batch_size=32, shuffle=True)

# Model parameters
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
max_seq_len = 128
num_epochs = 10
learning_rate = 1e-5

encoder = TransformerEncoder(vocab_size_src, d_model, num_heads, d_ff, num_layers, max_seq_len).to('cuda')
decoder = TransformerDecoder(vocab_size_tgt, d_model, num_heads, d_ff, num_layers, max_seq_len).to('cuda')

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.pad_token_id)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    
    for batch in dataloader:
        src = batch['src'].to('cuda')
        tgt = batch['tgt'].to('cuda')
        
        optimizer.zero_grad()
        
        # Encoder forward
        enc_output = encoder(src)
        
        # Create a shifted version of tgt for decoder input (i.e., start decoding with [BOS] token)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:].contiguous()
        
        # Create subsequent mask for the decoder
        tgt_mask = generate_subsequent_mask(tgt_input.size(1)).to('cuda')
        
        # Decoder forward
        logits = decoder(tgt_input, enc_output, mask=tgt_mask)
        
        # Reshape for loss computation
        logits = logits.view(-1, vocab_size_tgt)
        tgt_output = tgt_output.view(-1)
        
        # Loss computation
        loss = criterion(logits, tgt_output)
        loss.backward()
        
        # Clip gradients to avoid exploding gradient issue
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

def translate_sentence(model_encoder, model_decoder, src_sentence, max_length=50):
    encoder.eval()
    decoder.eval()
    
    # Tokenize source sentence
    src_tokens = src_tokenizer.encode(src_sentence, return_tensors='pt').to('cuda')
    
    # Forward through encoder
    enc_output = model_encoder(src_tokens)
    
    # Start decoding with [BOS] token
    tgt_tokens = torch.tensor([tgt_tokenizer.bos_token_id]).unsqueeze(0).to('cuda')
    
    for _ in range(max_length):
        # Create subsequent mask
        tgt_mask = generate_subsequent_mask(tgt_tokens.size(1)).to('cuda')
        
        # Forward through decoder
        logits = model_decoder(tgt_tokens, enc_output, tgt_mask)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        
        # Append predicted token
        tgt_tokens = torch.cat([tgt_tokens, next_token.unsqueeze(0)], dim=1)
        
        # Stop if end-of-sequence token is generated
        if next_token.item() == tgt_tokenizer.eos_token_id:
            break
    
    # Decode predicted sentence
    predicted_sentence = tgt_tokenizer.decode(tgt_tokens.squeeze().tolist(), skip_special_tokens=True)
    
    return predicted_sentence

# Test translation
src_sentence = "The weather is nice today."
translated_sentence = translate_sentence(encoder, decoder, src_sentence)
print(f"Translated sentence: {translated_sentence}")
