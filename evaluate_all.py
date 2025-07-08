import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision

# ← Ajusta estos paths según tu estructura
MODEL_INFO = {
    "CNN base": ("baseline_best.pth", "baseline"),
    "MobileNetV2": ("best_mobilenetv2.pth", "mobilenet_v2"),
    "ResNet18":    ("best_resnet18.pth",  "resnet18"),
    "EfficientNet": ("best_efficientnet.pth", "efficientnet_b0"),
}

NUM_CLASSES = 5
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = Path("dataset_split")

# Transform idéntico al de validación
val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    # Asegúrate de copiar tu clase Ensure3 o importarla
    transforms.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0]==1 else x[:3]),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# Dataset de prueba
test_ds = datasets.ImageFolder(DATA_ROOT/"test", val_tf)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
CLASS_NAMES = test_ds.classes  # ['author_001', ...]

# Carpeta de salida
out_dir = Path("results_confusion")
out_dir.mkdir(exist_ok=True)

for name, (ckpt, attr) in MODEL_INFO.items():
    # 1) Cargar modelo
    if attr == "baseline":
        from baseline_cnn import BaselineCNN
        model = BaselineCNN(num_classes=NUM_CLASSES)
    else:
        model = getattr(torchvision.models, attr)(weights="IMAGENET1K_V1")
        if attr=="mobilenet_v2":
            model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
        elif attr=="resnet18":
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif attr=="efficientnet_b0":
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.to(DEVICE).eval()

    # 2) Inferencia sobre test set
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(DEVICE)
            out = model(x).argmax(1).cpu().numpy()
            y_pred.extend(out)
            y_true.extend(y.numpy())

    # 3) Classification report CSV
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    df_report = pd.DataFrame(report).T
    df_report.to_csv(out_dir/f"{attr}_classification_report.csv", index=True)

    # 4) Matriz de confusión + heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
    plt.xlabel("Predicción")
    plt.ylabel("Verdadero")
    plt.title(f"Matriz de confusión ({name})")
    plt.tight_layout()
    plt.savefig(out_dir/f"{attr}_confusion_matrix.png")
    plt.close()

    print(f">>> {name}: report y matriz guardados.")
