# cross_validate.py

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torch import nn, optim
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import numpy as np

# ─── Helper to ensure 3-channel tensors ───────────────────────────────────────
class Ensure3:
    def __call__(self, x):
        c, h, w = x.shape
        if c == 1:
            return x.repeat(3, 1, 1)
        if c > 3:
            return x[:3, :, :]
        return x

# ─── Build a fresh MobileNet v2 for each fold ───────────────────────────────
def build_model(num_classes=5):
    model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

def main():
    # Paths & transforms
    data_root = Path("dataset_split")
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        Ensure3(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # Load the full training set (we'll split it ourselves)
    dataset = torchvision.datasets.ImageFolder(data_root/"train", transform=val_tf)
    X = np.array([s[0] for s in dataset.samples])  # file paths (unused directly)
    y = np.array(dataset.targets)

    # Prepare 5-fold stratified splits
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Loop over folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n=== Fold {fold} ===")

        # Create subsets and loaders
        train_subset = Subset(dataset, train_idx)
        val_subset   = Subset(dataset, val_idx)
        train_dl = DataLoader(train_subset, batch_size=8, shuffle=True,  num_workers=2)
        val_dl   = DataLoader(val_subset,   batch_size=8, shuffle=False, num_workers=2)

        # Instantiate model, loss, optimizer
        model     = build_model(num_classes=len(dataset.classes)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Simple early-stop training for this fold
        best_fold_acc, no_imp = 0.0, 0
        epochs_cv = 10
        for epoch in range(1, epochs_cv + 1):
            model.train()
            for x, yb in train_dl:
                x, yb = x.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), yb)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            correct = 0
            with torch.no_grad():
                for x, yb in val_dl:
                    x, yb = x.to(device), yb.to(device)
                    correct += (model(x).argmax(1) == yb).sum().item()
            val_acc = correct / len(val_subset)

            # Early stopping
            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= 3:
                    break

        print(f"Fold {fold} best val acc: {best_fold_acc:.3f}")
        fold_accs.append(best_fold_acc)

    # Summary
    print("\nFold accuracies:", [f"{acc:.3f}" for acc in fold_accs])
    print("Mean accuracy :", np.mean(fold_accs))

if __name__ == "__main__":
    main()
