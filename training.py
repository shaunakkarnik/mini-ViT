import torch
import torchvision
from torch import nn
from vision_transformer import VisionTransformer

device = torch.device('mps')

# training hyperparameters
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
if __name__ == "__main__":
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