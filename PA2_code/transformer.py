import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Head(nn.Module):
    """ one head (with optional local windowed attention for part3) """

    def __init__(self, n_embd, head_size, block_size, dropout, window_size, masked):
        super().__init__()
        self.map: Optional[torch.Tensor] = None
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.tril = torch.tril(torch.ones(block_size, block_size)).to(device)
        self.dropout = nn.Dropout(dropout)
        self.masked = masked
        self.window_size = window_size
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        wei = q @ k.transpose(-2,-1) * (C**-0.5) # (B, T, C) @ (B, C, T) -> (B, T, T)

        # Apply local windowed attention if window_size is specified
        if self.window_size is not None:
            # Create a window mask that only allows attention within the window size
            ones = torch.ones(T, T, device=device)
            window_mask = torch.triu(ones, diagonal=-self.window_size) * torch.tril(ones, diagonal=self.window_size)
            wei = wei * window_mask - 1e10 * (1 - window_mask)  # Apply mask by setting out-of-window values to large negative (masking out)

        # mask with tril, for the decoder
        if self.masked:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        self.map = wei
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads """

    def __init__(self, n_embd, n_head, head_size, block_size, dropout, window_size, masked):
        super().__init__()
        self.maps: Optional[torch.Tensor] = None
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout, window_size, masked) for _ in range(n_head)])
        self.proj = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        self.maps = torch.stack([h.map for h in self.heads], dim=0)
        return out   


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, dropout, window_size, masked):
        super().__init__()
        self.maps: Optional[torch.Tensor] = None
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout, window_size, masked)
        self.ffwd = FeedFoward(n_embd,dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)    
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        self.maps = self.sa.maps
        return x


class Encoder(nn.Module):
    """ Transformer Encoder """
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout, n_input, n_hidden, n_output):
        super().__init__()
        self.maps: Optional[torch.Tensor] = None
        self.block_size = block_size
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd).to(device)
        self.position_embedding_table = nn.Embedding(block_size, n_embd).to(device)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout, window_size = None, masked = False) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.clf = nn.Sequential(
            nn.Linear(n_input, n_hidden).to(device),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output).to(device),
        )

    def forward(self, idx, speaker_ids=None):
        idx = idx.to(device)
        if speaker_ids is not None:
            speaker_ids = speaker_ids.to(device)
        
        B, T = idx.shape

        # idx and speaker_ids are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        self.maps = torch.stack([b.maps for b in self.blocks]).view(-1, self.block_size, self.block_size)
        
        # global average pooling 
        # (B,T,C) -> (B,C)
        x = x.mean(dim=1) 

        logits = self.clf(x)

        if speaker_ids is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, speaker_ids)

        return logits, loss, self.maps
    

class Decoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.maps: Optional[torch.Tensor] = None
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout, window_size = None, masked = True) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        self.maps = torch.stack([b.maps for b in self.blocks]).view(-1, self.block_size, self.block_size)

        if targets is None:
            loss = None
        else:
            logits = logits.view(B*T, -1)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, self.maps
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx




class Decoder2(nn.Module):
    """ Decoder for part 3 with Local Windowed Attention"""
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout, window_size):
        super().__init__()
        self.maps: Optional[torch.Tensor] = None
        self.block_size = block_size
        self.window_size = window_size
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout, self.window_size, masked = True) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        self.maps = torch.stack([b.maps for b in self.blocks]).view(-1, self.block_size, self.block_size)

        if targets is None:
            loss = None
        else:
            logits = logits.view(B*T, -1)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, self.maps
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx



