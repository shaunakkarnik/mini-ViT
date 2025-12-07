import torch
import torchvision
from torch import nn
import math

# hyperparameters
batch_size = 32
num_workers = 2
embed_dim = 512
patch_dim = 4
num_heads = 6

image_size = 32
num_patches = (image_size // patch_dim) ** 2

device = torch.device('mps')

# load dataset and create dataloader
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers)


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
    
class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_embeddings = PatchEmbeddings()
        self.positional_encoding_table = nn.Embedding(num_patches + 1, embed_dim)
        self.self_attention_head = Head(embed_dim)

    def forward(self, x):
        patches = self.patch_embeddings(x)
        pos = self.positional_encoding_table(torch.arange(num_patches + 1, device=device))

        x = patches + pos

        x = self.self_attention_head(x)

        return x
