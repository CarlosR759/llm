# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset,DataLoader
import lightning as L


#Hyperparameters
batch_size = 64
block_size = 256 #Max context length for predictions


#Setting cuda if gpu is available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Read the data for traning.
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#the vocabulary of the model is made, after selecting all unique characters that happend in the text.
chars = sorted(list(set(text)))
vocab_size = len(chars)

#string to intenger
stoi = { ch:i for i,ch in enumerate(chars) }
#integer to string
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long).to(device)
mark = int(0.9*len(data)) # Mark separate the data in 90% - 10% ratio
train_data = data[:mark]
validation_data = data[mark:]

#Function for creating batches. Maybe we don't need it at all
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else validation_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


#In this part the data is separeted into a secuence of strings to train the model to make inference of the next token
X = []
Y = []

#This way of creating the data would work for small data sets. But check what lazy loading is.
for i in range(len(data) - block_size):
    X.append(data[i:i+block_size])
    Y.append(data[i+1:i+block_size+1])
X = torch.stack(X)
Y = torch.stack(Y)

X, Y = X.to(device), Y.to(device)
dataset = TensorDataset(X,Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %%
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=6, block_size=256):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList( [
                        nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
                        for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        token_embeddings = self.token_embedding(x)
        positional_embeddings = self.positional_embedding(torch.arange(seq_len, device=x.device))
        x = token_embeddings + positional_embeddings

        #creating masking
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        for block in self.blocks:
            x = block(x, x, tgt_mask=mask)

        x = self.ln_f(x)
        return self.lm_head(x)
# %%
model = DecoderOnlyTransformer(vocab_size=vocab_size, embed_dim=256, num_heads=8, num_layers=8)
model = model.to(device)

optimizer = Adam(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters of the model:  {total_params:,}")

# %%
for epoch in range(1):
    for x, y in dataloader:
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        print(f"Loss: {loss}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
