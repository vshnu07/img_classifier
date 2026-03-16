# Image Classification using HOG + Linear SVM

A machine learning pipeline for image classification using Histogram of Oriented Gradients (HOG) feature extraction and a Linear Support Vector Machine (SVM) classifier.

## Results

| Metric | Score |
|---|---|
| Training Accuracy | 90.45% |
| Testing Accuracy | 86.20% |
| Competition Ranking | Top 5% |

## Pipeline Overview

1. **Data Loading** — Load and preprocess raw image dataset
2. **Preprocessing** — Grayscale conversion and normalization
3. **Feature Extraction** — HOG (Histogram of Oriented Gradients) feature vectors
4. **Model Training** — Linear SVM with hyperparameter tuning
5. **Evaluation** — Accuracy on held-out test set

## Tech Stack

- Python
- NumPy
- Scikit-learn
- Matplotlib
- HOG Feature Extraction

## Key Highlights

- Achieved **86.20% test accuracy** with strong generalization on unseen data
- Engineered robust HOG feature vectors for improved model discrimination
- Optimized Linear SVM via rigorous hyperparameter fine-tuning
- Secured **top 5% ranking** in model performance competition

## Files

| File | Description |
|---|---|
| `Img_classifier.ipynb` | Full pipeline — data loading, HOG extraction, SVM training & evaluation |
| `img_classifer_lesser.ipynb` | Lightweight version with reduced feature set |

## How to Run
```bash
# Clone the repo
git clone https://github.com/vshnu07/img_classifier.git
cd img_classifier

# Install dependencies
pip install numpy scikit-learn matplotlib

# Open the notebook
jupyter notebook Img_classifier.ipynb
```

## Author

**Golla Vishnu** — [GitHub](https://github.com/vshnu07) · [LinkedIn](https://linkedin.com/in/golla-vishnu)
