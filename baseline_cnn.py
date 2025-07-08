import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Ensure3:
    def __call__(self, x):
        c, h, w = x.shape
        if c == 1:
            return x.repeat(3, 1, 1)
        if c > 3:
            return x[:3, :, :]
        return x

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    Ensure3(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

train_ds = datasets.ImageFolder("dataset_split/train", transform)
val_ds   = datasets.ImageFolder("dataset_split/val",   transform)
test_ds  = datasets.ImageFolder("dataset_split/test",  transform)

train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=0)
test_dl  = DataLoader(test_ds,  batch_size=8, shuffle=False, num_workers=0)

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BaselineCNN(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

epochs    = 30
patience  = 5
best_acc  = 0.0
no_improve= 0

history = {"epoch": [], "train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

for epoch in range(1, epochs+1):
    print(f"\nEpoch {epoch}/{epochs}")
    model.train()
    train_corr, train_loss = 0, 0.0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_corr += (out.argmax(1) == y).sum().item()
        train_loss += loss.item() * x.size(0)
    train_acc = train_corr / len(train_ds)
    train_loss /= len(train_ds)

    model.eval()
    val_corr, val_loss = 0, 0.0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            val_corr += (out.argmax(1) == y).sum().item()
            val_loss += loss.item() * x.size(0)
    val_acc = val_corr / len(val_ds)
    val_loss /= len(val_ds)

    print(f"train_acc: {train_acc:.3f} | val_acc: {val_acc:.3f}")
    print(f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

    history["epoch"].append(epoch)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "baseline_best.pth")
        no_improve = 0
        print("  📈 Saved new baseline checkpoint")
    else:
        no_improve += 1
        if no_improve >= patience:
            print("  ⏹️ Early stopping")
            break

pd.DataFrame(history).to_csv("baseline_history.csv", index=False)

model.load_state_dict(torch.load("baseline_best.pth", map_location=device))
model.eval()
test_corr = 0
with torch.no_grad():
    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        out = model(x)
        test_corr += (out.argmax(1) == y).sum().item()
test_acc = test_corr / len(test_ds)
print(f"\n🔬 Baseline CNN test accuracy: {test_acc:.3f}")
