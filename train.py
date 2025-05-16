import os
import sys
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import Model

#Hyperparameters
batch_size = 12
context_length = 16
max_iters = 200
learning_rate = 1e-3
eval_interval = 20
eval_iters = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

with open('data/订单商品名称.csv','r', encoding="utf-8") as f:
    text = f.read()

tokenizer = tiktoken.get_encoding("cl100k_base")
tokenized_text = tokenizer.encode(text)
tokenized_text = torch.tensor(data = tokenized_text, dtype=torch.long, device=device)

p_size = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:p_size]
valid_data = tokenized_text[p_size:]

model = Model().to(device)

def get_batch(split):
    # 选择数据集：训练集或验证集
    data = train_data if split == 'train' else valid_data
    
    # 随机采样 batch_size 个起始位置索引
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))

    # 构建输入序列 x 和目标序列 y
    x = torch.stack([data[idx:idx + context_length] for idx in idxs])
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs])

    return x, y

def estimate_loss():
    out = {}
    losses = torch.zeros(eval_iters)
    for split in ['train', 'valid']:
        for k in range(eval_iters):
            x,y = get_batch('train')
            _, loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for step in range(max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        print('Step:', step, 'Train Loss:', round(losses['train'].item(),3), 'Valid Loss:', round(losses['valid'].item(),3))
        
    x,y = get_batch('train')
    _, loss = model(x,y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


torch.save(model.state_dict(), 'model.ckpt')

