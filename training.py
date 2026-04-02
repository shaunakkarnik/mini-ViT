import torch
from torch import nn
from torch.utils.data import Subset
from torch.utils.data import default_collate
import torchvision
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image
import matplotlib.pyplot as plt
from vision_transformer import VisionTransformer

device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA")
else:
    device = torch.device('mps')
    print("Using MPS")

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

cutmix = v2.CutMix(alpha=0.2, num_classes=10)
mixup = v2.MixUp(alpha=0.2, num_classes=10)
cutmix_or_mixup = transforms.v2.RandomChoice([cutmix, mixup])

# instantiating twice to get different transforms on train vs. val
full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=train_transforms, download=True)
full_val_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=val_transforms, download=True)

train_size = int(len(full_train_dataset) * 0.9)

indices = torch.randperm(len(full_train_dataset))

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_val_dataset, val_indices)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=val_transforms, download=True)

batch_size = 512
num_workers = 4

def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)

# training hyperparameters
train_iters = 325
checkpoint_interval = 10
lr = 4e-3
weight_decay = 0.05
label_smoothing = 0.0
eps = 1e-8

model = VisionTransformer().to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer=optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=train_iters
)
scaler = torch.amp.GradScaler("cuda")

def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    num_correct = (preds == labels).sum().item()
    return num_correct

@torch.no_grad()
def evaluation(loader):

    model.eval()

    loss_sum = 0.0
    correct_preds = 0
    total = 0

    for batch, labels in loader:

        x = batch.to(device)
        targets = labels.to(device)

        with torch.amp.autocast("cuda"):
            out = model(x)
            loss = criterion(out, targets)

        loss_sum += loss.item()

        correct_preds += compute_accuracy(out, targets)
        total += targets.size(0)


    model.train()

    avg_loss = loss_sum / len(loader)
    avg_acc = correct_preds / total

    return avg_loss, avg_acc

save_weights_path = "vit_best_8.pth"
best_val_acc = 0.0

# main training loop
if __name__ == "__main__":
    for i in range(train_iters):
        train_losses_sum = 0.0

        model.train()

        for _, (batch, labels) in enumerate(train_loader):

            x = batch.to(device)
            targets = labels.to(device)

            with torch.amp.autocast("cuda"):
                out = model(x)
                loss = criterion(out, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_losses_sum += loss.item()

        avg_train_loss = train_losses_sum / len(train_loader)
        avg_val_loss, avg_val_acc = evaluation(val_loader)

        print(
            f"Epoch: {i + 1}"
            + f" Train Loss: {avg_train_loss:.4f}"
            + f" Val Loss: {avg_val_loss:.4f}"
            + f" Validation Accuracy: {avg_val_acc:.4f}"
        )

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), save_weights_path)

        if (i + 1) % checkpoint_interval == 0 or i == train_iters - 1:
            print(f"Best val acc: {best_val_acc:.4f}")