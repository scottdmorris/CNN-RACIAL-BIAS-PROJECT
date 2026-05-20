# Analyzing Racial Bias in Facial Recognition Algorithms

A deep learning project that trains and evaluates a facial recognition model to detect and quantify bias across demographic groups — using EfficientNet transfer learning, per-class metrics, and Grad-CAM visualizations.

**The goal is not to classify race.** The goal is to expose *bias disparity*: the unequal accuracy a model exhibits across demographic groups, mirroring failures documented in commercial systems used by law enforcement.

---

## Overview

Facial recognition systems are widely deployed but often perform unevenly across racial groups. This project investigates where and how those disparities emerge — even when the training data is deliberately balanced — and makes the model's decision-making visible through gradient-based heatmaps.

Originally built as a senior capstone project (Dec 2021) using a custom CNN. This version upgrades the architecture, scale, and analysis for a stronger portfolio demonstration.

---

## What Changed

| Original (Dec 2021) | This Version |
|---|---|
| Custom 2-layer CNN from scratch | EfficientNetB0 pretrained on ImageNet |
| ~7,000 training images | Full FairFace dataset (~86,000 train images) |
| 140-image test set | Full validation set (~13,000 images) |
| Accuracy + confusion matrix | Per-class F1 / precision / recall + Grad-CAM |
| Ran on local laptop (~1hr/run) | Google Colab (GPU) |

---

## Key Findings

- Accuracy varied significantly across all 7 racial groups even with a balanced dataset
- Per-class F1 scores revealed which groups were most and least affected by model errors
- Grad-CAM heatmaps showed whether the model attended to facial structure or surface-level cues like skin tone
- Consistent with NIST (2019) findings that most commercial facial recognition systems perform worse on non-white faces

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| TensorFlow / Keras | Model architecture, training, tf.data pipeline |
| EfficientNetB0 | Pretrained ImageNet backbone (transfer learning) |
| FairFace Dataset | Balanced, 7-class racial image dataset |
| NumPy / Pandas | Data loading and preprocessing |
| Matplotlib / Sklearn | Visualization and evaluation metrics |
| Grad-CAM | Gradient-based visual explanations |
| Google Colab | GPU training environment |

---

## Model Architecture

Transfer learning with EfficientNetB0:

1. **EfficientNetB0** (frozen) — pretrained on ImageNet; extracts facial features without training from scratch
2. **GlobalAveragePooling2D** — collapses the spatial feature map to a single vector
3. **Dropout (0.3)** — regularization to reduce overfitting
4. **Dense (7, softmax)** — outputs a probability for each of the 7 racial groups

Training runs in two phases: the EfficientNet base is frozen initially (only the head trains), then the top layers are unfrozen for fine-tuning at a reduced learning rate.

---

## Bias Analysis

Results are evaluated with:
- **Per-class precision, recall, and F1 score** — reveals which groups the model handles worst
- **Normalized confusion matrix** — shows which groups get confused with each other and whether errors are symmetric
- **Grad-CAM overlays** — visualizes what image regions the model attends to per class, revealing whether decisions are based on structural facial features or surface shortcuts

---

## Running the Notebook

The notebook is designed to run in **Google Colab** with FairFace downloaded via the Kaggle API.

1. Open `model.ipynb` in Google Colab
2. Run the setup cell — it will prompt you to upload your `kaggle.json` API key
3. FairFace (~3.5 GB) downloads once to your Google Drive and is reused across sessions
4. Run all cells top to bottom

Get a Kaggle API key at: kaggle.com → Settings → API → Create New Token

---

## References

- [FairFace Dataset](https://github.com/joojs/fairface) — Kärkkäinen & Joo, WACV 2021
- [Gender Shades](http://gendershades.org/) — Buolamwini & Gebru, 2018
- [NIST Face Recognition Vendor Test](https://www.nist.gov/programs-projects/face-recognition-vendor-testing-frvt) — 2019
