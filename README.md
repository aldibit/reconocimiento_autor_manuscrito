# Author recognition (English below)

## Conjunto de datos

Este repositorio contiene un conjunto de datos proveniente del proyecto 1000 people: https://github.com/Nexdata-AI/1000-People-Spanish-Handwriting-OCR-Data/blob/main/5.png
Las imágenes aquí utilizadas están disponibles de manera pública en el repositorio de Nexdata.

El conjunto de datos consta de muestras de escritura a mano:
- 5 autores distintos
- 20 muestras por cada autor
- 100 muestras en total

Las muestras se han recortado y etiquetado manualmente con el objetivo de entrenar una red neuronal para identificar o verificar autoría.

## Estructura

- `dataset/`: Contiene una subcarpeta por autor (por ejemplo, `author_001`, `author_002`, …). Cada subcarpeta contiene 20 muestras de imágenes recortadas.
- `cropped_images/`: Todas las imágenes recortadas e indexadas que se utilizaron para construir el conjunto de datos "dataset".
- `labels.csv`: Metadatos para todas las muestras. Incluye:
  - `filename`: nombre de archivo de la imagen
  - `person_id`: identificador de autoría
  - `expression_index`: índice de recorte (1-20)
  - `transcription`: vacío por el momento, a completar con la transcripción de la verdad fundamental, o *ground truth*, de lo escrito

## Cómo recrear el conjunto de datos

Correr el *script*
```bash
python crop_dataset.py
```
Este archivo:

- Lee los PNGs originales y las cajas de coordenadas especificadas manualmente en ```visualize_boxes.py```
- Recorta cada región
- Guarda las imágenes y las asigna a un autor en bins de 20
- Genera labels.csv con metadatos








# Author recognition - ENGLISH

## Dataset

This repository includes publicly available images from the project 1000 People: https://github.com/Nexdata-AI/1000-People-Spanish-Handwriting-OCR-Data/blob/main/5.png

The dataset holds handwritten image samples:
- 5 authors
- 20 samples per author
- 100 samples total

Samples have been manually cropped and labeled in order to train a neural network to identify or verify authorship of handwritten material

## Structure

- `dataset/`: Contains one subfolder per author (e.g., `author_001`, `author_002`, …), each with 20 cropped image samples.
- `cropped_images/`: All cropped images indexed and used to build the dataset.
- `labels.csv`: Metadata for all crops, including:
  - `filename`: Image filename
  - `person_id`: Person/author identifier
  - `expression_index`: Crop index (1–20)
  - `transcription`: Empty for now (can be annotated later)


## How to Recreate the Dataset

Run the script:

```bash
python crop_dataset.py
```

This script will:

- Read the original PNGs and manually defined box coordinates.
- Crop each region.
- Save the cropped images to cropped_images/.
- Assign each crop to an author in dataset/ using 20-sample bins.
- Generate labels.csv for metadata.
