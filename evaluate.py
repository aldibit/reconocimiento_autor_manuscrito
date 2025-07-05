import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Asegúrate de usar el mismo Ensure3
class Ensure3:
    def __call__(self, x):
        c, h, w = x.shape
        if c == 1:
            return x.repeat(3,1,1)
        if c > 3:
            return x[:3,:,:]
        return x

def main():
    data_root = Path("dataset_split")
    num_classes = 5

    # Transforms idénticos a los de validación
    val_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        Ensure3(),                                    # ← use this
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    test_ds = torchvision.datasets.ImageFolder(data_root/"test", val_tf)
    test_dl = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    # Modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load("best_mobilenetv2.pth", map_location=device))
    model.to(device).eval()

    # Recorrer test set
    all_preds = []
    all_labels= []
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    # Métricas
    labels = test_ds.classes  # autor_001, autor_002, …
    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=labels))
    print("=== Confusion Matrix ===")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

if __name__ == "__main__":
    main()
