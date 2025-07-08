import torch
import time
import pandas as pd
from ptflops import get_model_complexity_info
from torchvision import models
from baseline_cnn import BaselineCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5

def load_baseline():
    m = BaselineCNN(num_classes=NUM_CLASSES)
    m.load_state_dict(torch.load("baseline_best.pth", map_location=DEVICE))
    return m.to(DEVICE).eval()

def load_mobilenet():
    m = models.mobilenet_v2(weights="IMAGENET1K_V1")
    m.classifier[1] = torch.nn.Linear(m.last_channel, NUM_CLASSES)
    m.load_state_dict(torch.load("best_mobilenetv2.pth", map_location=DEVICE))
    return m.to(DEVICE).eval()

def load_resnet18():
    m = models.resnet18(weights="IMAGENET1K_V1")
    m.fc = torch.nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.load_state_dict(torch.load("best_resnet18.pth", map_location=DEVICE))
    return m.to(DEVICE).eval()

def load_efficientnet():
    m = models.efficientnet_b0(weights="IMAGENET1K_V1")
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    m.load_state_dict(torch.load("best_efficientnet.pth", map_location=DEVICE))
    return m.to(DEVICE).eval()

def measure(model, name):
    # Parámetros y FLOPs
    macs, params = get_model_complexity_info(
        model, (3,224,224), as_strings=False, print_per_layer_stat=False, verbose=False
    )
    # Inferencia
    x = torch.randn((8,3,224,224)).to(DEVICE)  # batch_size=8
    start = time.time()
    with torch.no_grad():
        model(x)
    end = time.time()
    t_ms = (end - start) * 1000

    return {
        "Modelo": name,
        "Parámetros (M)": params/1e6,
        "FLOPs (G)": macs/1e9,
        "Tiempo inferencia (ms/batch)": t_ms
    }

if __name__ == "__main__":
    runners = [
        ("Baseline CNN", load_baseline),
        ("MobileNetV2", load_mobilenet),
        ("ResNet18", load_resnet18),
        ("EfficientNet", load_efficientnet),
    ]
    results = []
    for name, loader in runners:
        print(f"Midiendo {name}...")
        m = loader()
        results.append(measure(m, name))
    df = pd.DataFrame(results)
    df.round(3).to_csv("efficiency_summary.csv", index=False)
    print("\n✅ Tabla de eficiencia guardada en efficiency_summary.csv:\n")
    print(df.round(3).to_string(index=False))
