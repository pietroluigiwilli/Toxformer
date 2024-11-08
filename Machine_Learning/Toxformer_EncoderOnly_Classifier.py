import torch 
import math

class Head(torch.nn.Module):
    def __init__(self, head_size, n_embd, dropout):
        super().__init__()
        self.key = torch.nn.Linear(int(n_embd), int(head_size), bias=False)
        self.query = torch.nn.Linear(int(n_embd), int(head_size), bias=False)
        self.value = torch.nn.Linear(int(n_embd), int(head_size), bias=False)
        self.dropout = torch.nn.Dropout(dropout) # fix prob

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = torch.nn.functional.softmax(wei,dim=-1) #softmax needs to be performed along the sequence axis (_,seq,_)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(torch.nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(int(head_size), int(n_embd), dropout) for _ in range(num_heads)])
        self.proj = torch.nn.Linear(int(n_embd), int(n_embd))
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #
        out = self.dropout(self.proj(out))
        return out

class FeedForward(torch.nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(int(n_embd), 4 * int(n_embd)),
            torch.nn.ReLU(), #investigate GeLU instead of ReLU
            torch.nn.Linear(4 * int(n_embd), int(n_embd)),
            torch.nn.Dropout(dropout), #dropout helps with overfitting
        )

    def forward(self, x):
        return self.net(x)

class Encoder(torch.nn.Module): 
    """Encoder block: communication followed by computation """

    def __init__(self, n_embd, n_head, head_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout)
        self.ffwd = FeedForward(int(n_embd), dropout)
        self.ln1 = torch.nn.LayerNorm(int(n_embd))
        self.ln2 = torch.nn.LayerNorm(int(n_embd)) #layernorm is not switched off at eval. so it is not buffered


    def forward(self, x):
        x = x + self.sa(self.ln1(x)) #in residual connection, addition distributes gradients equally to all of its branches.
        x = x + self.ffwd(self.ln2(x)) #LayerNorm before the transformation instead of after avoids the instability of the training due to large output gradients
        return x

class PositionalEncoding(torch.nn.Module):
    def __init__(self, n_embd, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, n_embd)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1),:] 
        return x

class ClassificationTransformer(torch.nn.Module):

    def __init__(self,n_embd,n_head,n_layers,max_map,seq_len,dropout,n_class):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        head_size = int(n_embd) // n_head
        self.token_embedder = torch.nn.Embedding(int(max_map+1), n_embd)
        self.position_embedder = PositionalEncoding(n_embd,dropout)
        self.EncoderBlock = torch.nn.Sequential(*[Encoder(n_embd=n_embd, n_head=n_head, head_size=head_size, dropout=dropout) for _ in range(n_layers)])
        self.out_map = torch.nn.Linear(seq_len*n_embd,n_class)
        self.softmax = torch.nn.Softmax()


    def forward(self, x):
        B, T = x.shape
        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedder(x.long())
        x = self.position_embedder(token_emb) # (B,T,C)

        x = self.EncoderBlock(x) # (B,T,C)

        x = torch.flatten(x,start_dim=-2, end_dim=-1)
        y = self.out_map(x)
        y = self.softmax(y)
        return y
