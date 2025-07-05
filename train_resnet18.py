import os, copy, torch, torchvision
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm.auto import tqdm

# ---------- Named helper (picklable) ----------
class Ensure3:
    """Return a 3‑channel tensor.
       • If C==1  → repeat to 3
       • If C==3  → return as‑is
       • If C>3   → take first 3 channels
    """
    def __call__(self, x):
        c, h, w = x.shape
        if c == 1:
            return x.repeat(3, 1, 1)
        if c > 3:
            return x[:3, :, :]
        return x
# ---------- Pipeline wrapped in main() ----------
def main():
    data_root = Path("dataset_split")
    model_out = "best_mobilenetv2.pth"
    num_classes = 5
    batch_size  = 8
    epochs      = 40
    patience    = 5

    train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=.2, contrast=.2),
    transforms.ToTensor(),
    Ensure3(),                                    # ← use this
    transforms.Normalize([0.5]*3, [0.5]*3),
])

    val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    Ensure3(),                                    # ← use this
    transforms.Normalize([0.5]*3, [0.5]*3),
])
    train_ds = torchvision.datasets.ImageFolder(data_root / "train", train_tf)
    val_ds   = torchvision.datasets.ImageFolder(data_root / "val",   val_tf)
    test_ds  = torchvision.datasets.ImageFolder(data_root / "test",  val_tf)

    train_dl = DataLoader(train_ds, batch_size, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size, shuffle=False, num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_wts, best_acc, no_imp = copy.deepcopy(model.state_dict()), 0.0, 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        # ----- training -----
        model.train()
        train_correct = 0
        for x, y in tqdm(train_dl, desc="train"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_correct += (out.argmax(1) == y).sum().item()

        train_acc = train_correct / len(train_ds)

        # ----- validation -----
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_correct += (out.argmax(1) == y).sum().item()
        val_acc = val_correct / len(val_ds)
        print(f"train acc: {train_acc:.3f} | val acc: {val_acc:.3f}")

        # ----- early‑stop -----
        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "best_resnet18.pth")
            no_imp = 0
            print("  📈 Saved new best model")
        else:
            no_imp += 1
            if no_imp >= patience:
                print("  ⏹️  Early stopping")
                break

    # ----- test -----
    model.load_state_dict(best_wts)
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            test_correct += (out.argmax(1) == y).sum().item()
    test_acc = test_correct / len(test_ds)
    print(f"\n🔬 Test accuracy: {test_acc:.3f}")

# ---------- Windows‑safe entry ----------
if __name__ == "__main__":
    main()