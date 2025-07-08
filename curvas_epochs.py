import pandas as pd
import matplotlib.pyplot as plt

# Cargar los archivos CSV
baseline = pd.read_csv('baseline_history.csv')
mobilenet = pd.read_csv('mobilenet_history.csv')
resnet = pd.read_csv('resnet_history.csv')
efficientnet = pd.read_csv('efficientnet_history.csv')

# Crear gráficas de precisión
plt.figure(figsize=(14, 6))

# Precisión de entrenamiento
plt.subplot(1, 2, 1)
plt.plot(baseline['epoch'], baseline['train_acc'], label='CNN base')
plt.plot(mobilenet['epoch'], mobilenet['train_acc'], label='MobileNetV2')
plt.plot(resnet['epoch'], resnet['train_acc'], label='ResNet18')
plt.plot(efficientnet['epoch'], efficientnet['train_acc'], label='EfficientNet')
plt.title('Precisión de Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)

# Precisión de validación
plt.subplot(1, 2, 2)
plt.plot(baseline['epoch'], baseline['val_acc'], label='CNN base')
plt.plot(mobilenet['epoch'], mobilenet['val_acc'], label='MobileNetV2')
plt.plot(resnet['epoch'], resnet['val_acc'], label='ResNet18')
plt.plot(efficientnet['epoch'], efficientnet['val_acc'], label='EfficientNet-B0')
plt.title('Precisión de Validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Crear gráficas de pérdida
plt.figure(figsize=(14, 6))

# Pérdida de entrenamiento
plt.subplot(1, 2, 1)
plt.plot(baseline['epoch'], baseline['train_loss'], label='CNN base')
plt.plot(mobilenet['epoch'], mobilenet['train_loss'], label='MobileNetV2')
plt.plot(resnet['epoch'], resnet['train_loss'], label='ResNet18')
plt.plot(efficientnet['epoch'], efficientnet['train_loss'], label='EfficientNet-B0')
plt.title('Pérdida de Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)

# Pérdida de validación
plt.subplot(1, 2, 2)
plt.plot(baseline['epoch'], baseline['val_loss'], label='CNN base')
plt.plot(mobilenet['epoch'], mobilenet['val_loss'], label='MobileNetV2')
plt.plot(resnet['epoch'], resnet['val_loss'], label='ResNet18')
plt.plot(efficientnet['epoch'], efficientnet['val_loss'], label='EfficientNet-B0')
plt.title('Pérdida de Validación')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
