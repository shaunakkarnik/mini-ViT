import torch
import torchvision
from torch import nn
import math

# model hyperparameters
embed_dim = 128
patch_dim = 4
num_heads = 8 # random, picked from Andrej Karpathy video

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
    
class Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.MHA = MultiHeadAttention(num_heads)
        self.mlp = nn.Sequential (
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, x):

        x_prime = self.norm(x)
        x_prime = self.MHA(x_prime)
        x_prime = x + x_prime

        out = self.norm(x_prime)
        out = self.mlp(out)
        out = out + x_prime

        return out

class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.mlp = nn.Sequential (
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 10)
        )

    def forward(self, x):
        out = self.mlp(x)
        return out

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_embeddings = PatchEmbeddings()
        self.positional_encoding_table = nn.Embedding(num_patches + 1, embed_dim)
        self.blocks = nn.Sequential(
            Block(),
            Block(),
            Block()
        )
        self.classification_head = ClassificationHead()

    def forward(self, x):
        patches = self.patch_embeddings(x)
        pos = self.positional_encoding_table(torch.arange(num_patches + 1, device=device))

        x = patches + pos

        x = self.blocks(x)

        class_token = x[:, 0, :]

        out = self.classification_head(class_token)

        return out