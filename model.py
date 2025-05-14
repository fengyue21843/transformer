import torch
import torch.nn as nn
from torch.nn import functional as F
import math

#Hyperparameters
batch_size = 4
d_model = 512
context_length = 16
num_heads = 8
head_size = d_model //num_heads
num_blocks = 12
dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FeedForward(nn.Module):
    def __init__(self):
        super(). __init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        )

    
    def forward(self,x):
        return self.ffn(x)
    

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(d_model, head_size, bias = False)
        self.Wk = nn.Linear(d_model, head_size, bias = False)
        self.Wv = nn.Linear(d_model, head_size, bias = False)
        self.register_buffer('mask',torch.tril(torch.ones(context_length, context_length)))
        self.Dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, D = x.shape
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        output = (q @ k.transpose(-2,-1))/math.sqrt(head_size)
        output = output = output.masked_fill(self.mask[:T,:T]==0, float('-inf'))
        output = F.softmax(output, dim = -1)
        output = self.Dropout(output) #Optional dropout
        output = output @ v

        return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Attention() for _ in range(num_heads)])
        self.Wo = nn.Linear(num_heads * head_size, d_model)
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.Dropout(self.Wo(output)) #先进行线性变换再进行dropout

        return output
    
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention()
        self.ffn = FeedForward()
    
    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x

class Model(nn.Module):
    def __init__(self, max_token_value = 1000256):
        super().__init__()
        self.vocab_linear = nn.Linear(d_model, max_token_value)
        self.te_lookup_table = nn.Embedding(max_token_value, d_model) #token_embedding
        self.transformer_block = nn.Sequential(
            *([TransformerBlock() for _ in range(num_blocks)] + [nn.LayerNorm(d_model)])
        )
    def forward(self, x_batch, y_batch = None):
        B,T = x_batch.shape
        pe_lookup_table = torch.zeros(context_length, d_model, device=device)
        position = torch.arange(0, context_length, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2).float() / d_model)
        pe_lookup_table[:,0::2] = torch.sin(position * div_term)
        pe_lookup_table[:,1::2] = torch.cos(position * div_term)

        output = self.te_lookup_table(x_batch) + pe_lookup_table
        output = self.transformer_block(output)
        logits = self.vocab_linear(output)

        if y_batch is not None:
            B,T,D = logits.shape
            logits_reshaped = logits.view(B*T, D)
            y_batch_reshaped = y_batch.view(B*T)
            loss = F.cross_entropy(input = logits_reshaped, target = y_batch_reshaped)
            
        else:
            loss = None
        return logits, loss

if __name__ == "__main__":
    torch.manual_seed(0)

    vocab_size = 1000  # 假设词表大小为 1000
    model = Model(max_token_value=vocab_size).to(device)

    # 构造一个假的输入 batch：batch_size=4，sequence_length=16
    x_batch = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long).to(device)
    y_batch = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long).to(device)

    # 测试 forward
    logits, loss = model(x_batch, y_batch)

    print("Logits shape:", logits.shape)   # 应为 (4, 16, vocab_size)
    print("Loss:", loss.item())




