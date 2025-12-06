import torch
import torchvision
from torch import nn

# hyperparameters
batch_size = 32
num_workers = 2
embed_dim = 512
patch_dim = 4

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
        patches = self.unfold(x) # B, C * P^2, N
        patches = patches.permute(0, 2, 1) # B, N, C * P^2
        patches = self.projection(patches) # B, N, embed_dim

        B, _, _ = patches.shape

        class_token = self.cls.expand(B, -1, -1) # B, 1, embed_dim

        patch_embeddings = torch.cat((class_token, patches), dim=1)

        return patch_embeddings
    
class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_embeddings = PatchEmbeddings()
        self.positional_encoding_table = nn.Embedding(num_patches + 1, embed_dim)

    def forward(self, x):
        patches = self.patch_embeddings(x)
        pos = self.positional_encoding_table(torch.arange(num_patches + 1, device=device))

        x = patches + pos
        return x
