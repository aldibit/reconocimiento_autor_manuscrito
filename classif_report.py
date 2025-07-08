import pandas as pd

# Rutas a los archivos CSV
files = {
    "Baseline CNN":      "results_confusion/baseline_classification_report.csv",
    "MobileNetV2":       "results_confusion/mobilenet_v2_classification_report.csv",
    "ResNet18":          "results_confusion/resnet18_classification_report.csv",
    "EfficientNet":      "results_confusion/efficientnet_b0_classification_report.csv",
}

# Leer y mostrar tablas
for name, path in files.items():
    df = pd.read_csv(path, index_col=0)
    # Filtrar solo los autores
    df_authors = df.loc[["author_001","author_002","author_003","author_004","author_005"], 
                        ["precision","recall","f1-score"]]
    print(f"\n**{name}**\n")
    print(df_authors.round(2))
