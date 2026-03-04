# Analyzing Racial Bias in Facial Recognition Algorithms

A deep learning project that trains and evaluates a Convolutional Neural Network (CNN) to detect and quantify racial bias across demographic groups in facial recognition systems.

## Overview

Facial recognition systems are widely deployed but often perform unevenly across racial and gender groups. This project investigates *why* those disparities exist and *how significant* they are, using a balanced, real-world dataset and measurable accuracy metrics across subgroups.

## Key Findings

- Identified statistically significant accuracy disparities across racial demographic groups
- Visualized per-group performance gaps using confusion matrices and accuracy plots
- Demonstrated that training data diversity directly impacts model fairness
- Highlighted ethical implications for deploying facial recognition in high-stakes settings

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| TensorFlow / Keras | CNN architecture and training |
| FairFace Dataset | Balanced, multi-racial image dataset (UCLA) |
| NumPy / Pandas | Data preprocessing and manipulation |
| Matplotlib / Sklearn | Results visualization and evaluation metrics |

## Model Architecture

- Convolutional Neural Network (CNN) trained on the FairFace dataset
- FairFace contains balanced racial classifications across 7 groups: White, Black, Latino/Hispanic, East Asian, Southeast Asian, Middle Eastern, and Indian
- Evaluated using per-class accuracy, precision, recall, and F1 score

## Results

Accuracy varied meaningfully across demographic groups, with some groups showing notably lower recognition rates — consistent with bias findings in published research on commercial facial recognition systems.

## Why This Matters

As AI systems become embedded in hiring, law enforcement, and healthcare, understanding and mitigating demographic bias is critical. This project demonstrates the importance of balanced training data and rigorous fairness evaluation before deployment.

## Setup
```bash
git clone https://github.com/scottdmorris/[repo-name]
cd [repo-name]
pip install -r requirements.txt
jupyter notebook
```

## References

- [FairFace Dataset](https://github.com/joojs/fairface) — Kärkkäinen & Joo, 2021
- [Gender Shades](http://gendershades.org/) — Buolamwini & Gebru, 2018
