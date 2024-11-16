# 🤖Machine Learning Classifier⚙️

A Python implementation of handwritten digit recognition using Support Vector Machine (SVM) and Principal Component Analysis (PCA).

Note: Due to privacy policies, I am not allowed to post the dataset publicly.

---

## Table of Contents📋
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Implementation](#implementation)
- [Running the Code](#running-the-code)
- [Experimentation](#experimentation)

---

## Overview📝
This project implements a machine learning pipeline for recognizing handwritten digits using the following techniques:
- Dimensionality reduction with PCA
- Classification using linear SVM
- Performance evaluation on validation set

---

## Dependencies🛠️
```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn import svm

```

---

## Dataset📂
The project uses the Digits Dataset from scikit-learn, which contains:
- Handwritten digit images (0-9)
- Features extracted from the images
- Target labels indicating the digit

---

## Implementation💻

### 1. Data Loading
```python
digits = load_digits()
x, y = digits.data, digits.target
```

### 2. Dimensionality Reduction
```python
pca = PCA(n_components=8)
pca.fit(x)
x = pca.transform(x)
```

### 3. Model Training and Prediction
```python
svc = svm.SVC(kernel='linear')
svc.fit(x_train, y_train)
y_predicted = svc.predict(x_valid)
```

---

## Running the Code▶️
1. Load the digits dataset
2. Apply PCA transformation
3. Split data into training and validation sets
4. Train SVM classifier
5. Make predictions and evaluate performance

---

## Experimentation⚗️
The following parameters can be modified to optimize performance:
- Number of PCA components
- Training set size
- SVM kernel and parameters

Learning curves can be plotted to visualize the impact of:
- Number of PCA components vs. accuracy
- Training set size vs. accuracy