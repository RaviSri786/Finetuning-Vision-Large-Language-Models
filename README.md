# 📊 Fine-Tuning Vision-Language Models

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

🔗 [Hugging Face Dataset Link](https://huggingface.co/datasets/ravisri/bird-presence-classification)

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

🔗 [ravisri/clip-bird-detector](https://huggingface.co/ravisri/clip-bird-detector)

### 2. **BLIP (Bootstrapped Language-Image Pre-training)**
- Fine-tuned for binary classification.
- Showed promising results but had imbalanced class predictions.

🔗 [ravisri/blip-bird-classifier](https://huggingface.co/ravisri/blip-bird-classifier)

### 3. **LLaMA (Large Language Model pre-trained on text data)**
- Adapted for vision-language tasks using multimodal extensions.
- Performed moderately compared to CLIP and BLIP.

🔗 [ravisri/finetuned-llama-model](https://huggingface.co/ravisri/finetuned-llama-model)

---

## 🛠️ Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ravisri/FineTuning-Vision-Large-Language-Models.git
```

### 2. Install Dependencies

Ensure Python is installed, then install the required libraries:

```bash
pip install torch torchvision transformers datasets scikit-learn pandas matplotlib seaborn
```

### 3. Access the Dataset

Download the dataset from Hugging Face:

```python
from datasets import load_dataset
dataset = load_dataset("ravisri/bird-presence-classification")
```

### 4. Run the Notebooks

Open the following Jupyter notebooks for training and evaluation:
- `CLIP_finetuning.ipynb`
- `BLIP_finetuning.ipynb`
- `LLaMA_finetuning.ipynb`

### 5. GPU Configuration

The models were trained on a **NVIDIA SXM4 A100** GPU rented from **Vast.ai** at $1/hour. You can replicate the setup using similar GPU instances or local hardware.

---

## 📈 Results

### **CLIP**
- **Accuracy**: 88.74%
- **F1-Score**: 0.8893
- **Precision**: 0.9173 (Bird), 0.8576 (No_bird)
- **Recall**: 0.8472 (Bird), 0.9233 (No_bird)

### **BLIP**
- **Accuracy**: 89.74%
- **F1-Score**: 0.6533
- **Precision**: 0.1818 (Bird), 0.4929 (No_bird)
- **Recall**: 0.0069 (Bird), 0.9686 (No_bird)

### **LLaMA**
- **Accuracy**: 85.74%
- **F1-Score**: 0.8536 (Bird), 0.8610 (No_bird)
- **Precision**: 0.8787 (Bird), 0.8383 (No_bird)
- **Recall**: 0.8299 (Bird), 0.8850 (No_bird)

✅ **Conclusion**: **BLIP** achieved the highest accuracy (89.74%) among the three models, followed closely by **CLIP** (88.74%).

---

## 📁 Files & Structure

```
FineTuning-Vision-Large-Language-Models/
├── notebooks/
│   ├── CLIP_finetuning.ipynb
│   ├── BLIP_finetuning.ipynb
│   └── LLaMA_finetuning.ipynb
├── data/
│   └── bird_presence_classification.csv
├── models/
│   ├── clip_model.pth
│   ├── blip_model.pth
│   └── llama_model.pth
├── results/
│   ├── evaluation_metrics.csv
│   └── test_predictions.csv
└── README.md
```

- **`notebooks/`**: Contains Jupyter notebooks for fine-tuning and evaluation.
- **`data/`**: Includes the raw dataset files.
- **`models/`**: Saves the fine-tuned model checkpoints.
- **`results/`**: Stores evaluation metrics and prediction outputs.

---

## 👩‍💻 Contributing

If you'd like to contribute to this project, feel free to open issues or submit pull requests. Contributions such as:
- Improving model performance
- Adding new models for comparison
- Enhancing documentation

are highly appreciated!

---

## 📞 Contact

For any questions or feedback, please reach out:
📧 **Email**: ravisripallam@gmail.com  
🔗 **LinkedIn**: [Ravi Sri Pallam](https://www.linkedin.com/in/ravisripallam786/)  
🐙 **GitHub**: [ravisri](https://github.com/RaviSri786)

---

## 🚀 Acknowledgments

- **Hugging Face**: For providing the dataset and model hosting platform.
- **Vast.ai**: For the high-performance GPU instances used for training.
- **Open Source Community**: For the libraries and frameworks that made this project possible.
```
