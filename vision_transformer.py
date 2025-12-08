import torch
import torchvision
from torch import nn
import math

# model hyperparameters
embed_dim = 512
patch_dim = 4
num_heads = 8 # random, picked from Andrej Karpathy video
classifier_hidden_dim = 1024

image_size = 32
num_patches = (image_size // patch_dim) ** 2

device = torch.device('mps')

class PatchEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()

        self.unfold = nn.Unfold(kernel_size=patch_dim, stride=patch_dim)
        self.projection = nn.Linear((patch_dim ** 2) * 3, embed_dim)
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim)) # 1, 1, embed_dim

    def forward(self, x):
        # creates patches of size patch_dim x patch_dim
        patches = self.unfold(x) # B, C * P^2, N
        patches = patches.permute(0, 2, 1) # B, N, C * P^2

        # creates patch embeddings
        patches = self.projection(patches) # B, N, embed_dim

        B, _, _ = patches.shape

        # creates class token
        class_token = self.cls.expand(B, -1, -1) # B, 1, embed_dim

        patch_embeddings = torch.cat((class_token, patches), dim=1) # B, N + 1, embed_dim

        return patch_embeddings
    
class Head(nn.Module):
    def __init__(self, head_dim):
        super().__init__()

        self.query = nn.Linear(embed_dim, head_dim)
        self.key = nn.Linear(embed_dim, head_dim)
        self.value = nn.Linear(embed_dim, head_dim)

    def forward(self, x):
        q = self.query(x) # B, N+1, head_dim
        k = self.query(x) # B, N+1, head_dim
        v = self.query(x) # B, N+1, head_dim

        _, _, d_k = q.shape

        # scaled dot-product attention
        weights = q @ k.transpose(1, 2) # B, N+1, N+1 --> affinities between patches
        weights = weights / math.sqrt(d_k)
        weights = torch.softmax(weights, dim=1)

        attention = weights @ v # B, N+1, head_dim

        return attention
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        
        # list of Heads of head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(Head(embed_dim // num_heads))

        # output projection
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        single_head_outs = []
        for head in self.heads:
            single_head_outs.append(head(x)) # B, N+1, head_dim

        # concatenate output of heads
        multi_head_out = torch.cat(single_head_outs, dim=2) # B, N+1, embed_dim

        out = self.projection(multi_head_out) # B, N+1, embed_dim

        return out

class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.mlp = nn.Sequential (
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x):
        out = self.mlp(x)
        return out

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_embeddings = PatchEmbeddings()
        self.positional_encoding_table = nn.Embedding(num_patches + 1, embed_dim)
        self.MHA = MultiHeadAttention(num_heads)
        self.classification_head = ClassificationHead(classifier_hidden_dim)

    def forward(self, x):
        patches = self.patch_embeddings(x)
        pos = self.positional_encoding_table(torch.arange(num_patches + 1, device=device))

        x = patches + pos

        x = self.MHA(x)

        class_token = x[:, 0, :]

        out = self.classification_head(class_token)

        return out
    

# --- training ----

# # training hyperparameters
batch_size = 32
num_workers = 2 # ?
train_iters = 2
eval_interval = 1
eval_iters = 2

# load full CIFAR-10 dataset
full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)

# split train into train and val
train_size = int(len(full_train_dataset) * 0.9)
val_size = len(full_train_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

# create dataloaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=2)

model = VisionTransformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# evaluation func to get avg train and val loss over eval_iters number of batches
@torch.no_grad()
def eval(iteration):

    train_losses = torch.zeros(eval_iters)
    for i, (batch, labels) in enumerate(train_loader):
        if i == eval_iters:
            break

        x = batch.to(device)
        targets = labels.to(device)
        out = model(x)
        loss = criterion(out, targets)
        train_losses[i] = loss.item()

    avg_train_loss = train_losses.mean(dim=0)

    val_losses = torch.zeros(eval_iters)
    for i, (batch, labels) in enumerate(val_loader):
        if i == eval_iters:
            break

        x = batch.to(device)
        targets = labels.to(device)
        out = model(x)
        loss = criterion(out, targets)
        val_losses[i] = loss.item()

    avg_val_loss = val_losses.mean(dim=0)

    print("Iteration " + str(iteration) + " - Train loss: " + str(avg_train_loss.item()) + ", Val loss: " + str(avg_val_loss.item()))

# main training loop
for i in range(train_iters):

    if i % eval_interval == 0:
        eval(i)

    for batch, labels in train_loader:
        
        x = batch.to(device)
        targets = labels.to(device)

        out = model(x)

        loss = criterion(out, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



