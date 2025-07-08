import os, copy, torch, torchvision
import pandas as pd
import time
import torch
from ptflops import get_model_complexity_info
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm.auto import tqdm
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ---------- Named helper (picklable) ----------
class Ensure3:
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

    model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_wts, best_acc, no_imp = copy.deepcopy(model.state_dict()), 0.0, 0
    history = {"epoch": [], "train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        # ----- training -----
        model.train()
        train_correct = 0
        train_loss = 0.0
        for x, y in tqdm(train_dl, desc="train"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_correct += (out.argmax(1) == y).sum().item()
            train_loss += loss.item() * x.size(0)

        train_acc = train_correct / len(train_ds)
        train_loss /= len(train_ds)
        # ----- validation -----
        model.eval()
        val_correct = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)

                val_correct += (out.argmax(1) == y).sum().item()
                val_loss += loss.item() * x.size(0)

        val_acc = val_correct / len(val_ds)
        val_loss /= len(val_ds)

        print(f"train acc: {train_acc:.3f} | val acc: {val_acc:.3f}")
        print(f"train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        # Guardar historial
        history["epoch"].append(epoch)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # ----- early‑stop -----
        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, model_out)
            no_imp = 0
            print("  📈 Saved new best mobilenet model")
        else:
            no_imp += 1
            if no_imp >= patience:
                print("  ⏹️  Early stopping")
                break

    pd.DataFrame(history).to_csv("mobilenet_history.csv", index=False)
    
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

    # medimos eficiencia del modelo MobileNetV2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar nuevamente el modelo entrenado (asegúrate que coincida con model_out)
    num_classes = 5
    model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load("best_mobilenetv2.pth", map_location=device))
    model.to(device)
    model.eval()

    # Calcular FLOPs y parámetros
    with torch.no_grad():
        macs, params = get_model_complexity_info(
            model, (3, 224, 224), as_strings=True,
            print_per_layer_stat=False, verbose=False
        )
        print(f"Complejidad: {macs}, Parámetros: {params}")

    # Calcular tiempo de inferencia
    x_dummy = torch.randn((8, 3, 224, 224)).to(device)  # batch_size=8
    start = time.time()
    with torch.no_grad():
        model(x_dummy)
    end = time.time()

    inference_time = (end - start) * 1000  # en milisegundos
    print(f"Tiempo inferencia (ms/batch): {inference_time:.2f}")