# Finetuning-Vision-Large-Language-Models

This repository contains the code and results of fine-tuning three vision-language models — **CLIP**, **BLIP**, and **LLaMA** — for binary bird classification (bird vs. no-bird). The models were trained on a custom dataset uploaded to Hugging Face, and the best-performing model was identified through comprehensive evaluation metrics.

## 📚 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Environment Setup](#environment-setup)
- [Results](#results)
- [Files & Structure](#files--structure)
- [Contributing](#contributing)
- [Contact](#contact)

---

## 🌟 Overview

The goal of this project is to evaluate the performance of three popular vision-language models on a custom binary classification task: identifying whether an image contains a bird or not. The models used are:

1. **CLIP (Contrastive Language–Image Pre-training)**
2. **BLIP (Bootstrapped Language-Image Pre-training)**
3. **LLaMA (Large Language Model pre-trained on text data)**

Each model was fine-tuned using a custom dataset of images labeled as "bird" or "no-bird." The training was performed on a high-performance GPU instance (`NVIDIA SXM4 A100`) rented from **Vast.ai** at $1/hour, ensuring efficient computation.

The fine-tuned models and their evaluation results are saved to Hugging Face, and all Jupyter notebooks containing the training and evaluation code are available in this repository.

---

## 📊 Dataset

The dataset used for this project is a custom binary classification dataset named **Bird Presence Classification**, which consists of two classes:
- **Bird**: Images containing birds.
- **No_bird**: Images without birds.

The dataset is publicly available on Hugging Face:
🔗 [ravisri/bird-presence-classification](https://huggingface.co/datasets/ravisri/bird-presence-classification )

### Key details about the dataset:
- Total samples: 2317 for training, 575 for testing
- Class distribution:
  - Bird: 1159 for training, 288 for testing
  - No_bird: 1158 for training, 287 for testing

The dataset was split into training (80%) and testing (20%) sets for model evaluation.

---

## 🏗️ Models

Three vision-language models were fine-tuned for this task:

### 1. **CLIP (Contrastive Language–Image Pre-training)**
- Trained with a linear head for binary classification.
- Achieved high accuracy and strong balanced performance.

### 2. **BLIP (Bootstrapped Language-Image Pre-training)**
- Fine-tuned for binary classification.
- Showed promising results but had imbalanced class predictions.

### 3. **LLaMA (Large Language Model pre-trained on text data)**
- Adapted for vision-language tasks using multimodal extensions.
- Performed moderately compared to CLIP and BLIP.

All fine-tuned models are saved to Hugging Face:
- 🔗 [CLIP - ravisri/clip-bird-detector](https://huggingface.co/ravisri/clip-bird-detector )
- 🔗 [BLIP - ravisri/blip-bird-classifier](https://huggingface.co/ravisri/blip-bird-classifier )
- 🔗 [LLaMA - ravisri/finetuned-llama-model](https://huggingface.co/ravisri/finetuned-llama-model )

---

## 🛠️ Environment Setup

To replicate the experiments, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/ravisri/FineTuning-Vision-Large-Language-Models.git 
