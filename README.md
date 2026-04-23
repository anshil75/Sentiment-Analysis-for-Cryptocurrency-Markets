# Sentiment Analysis of the Cryptocurrency Market using Machine Learning

This project analyzes the relationship between public sentiment and cryptocurrency price movements using real-world data.

## 📌 Overview

We collected and combined cryptocurrency-related data from multiple sources, including news headlines, social media (tweets), and Bitcoin price data (2018–2023). The dataset consists of over 240,000 records.

## 🎯 Objective

To classify the sentiment of crypto-related text (positive / neutral / negative) and understand how market sentiment correlates with cryptocurrency price trends.

## 🛠️ Tech Stack

* Python 3.10+
* Pandas, NumPy, SciPy
* Scikit-learn
* NLTK (Natural Language Processing)
* Matplotlib, Seaborn, WordCloud

## ⚙️ Features

* Label cleaning and text preprocessing (lowercasing, stopwords, lemmatisation, slang normalisation)
* TF-IDF feature extraction (unigrams + bigrams, fit on training data only)
* Stratified 80/20 train/test split
* Hyperparameter tuning via cross-validation on training data (no test-set leakage)
* Five classifiers: Logistic Regression, SVM (LinearSVC), Multinomial Naive Bayes, KNN, Decision Tree
* Evaluation: accuracy, weighted precision/recall/F1, confusion matrices, 5-fold CV

## 📁 Dataset

Data sourced from Kaggle (2018–2023 crypto data).  
**The dataset file is not committed to this repository.**  
Download `master_crypto_sentiment_CLEAN.csv` from Kaggle and place it in the `data/` folder, then update the `PATH` variable in `Preprocessing_Model_Training.ipynb`.

## 🚀 Getting Started

```bash
pip install -r requirements.txt
jupyter lab Preprocessing_Model_Training.ipynb
```

Run notebooks in order:
1. `02_Dataset_Study_EDA.ipynb` — exploratory data analysis
2. `Preprocessing_Model_Training.ipynb` — preprocessing, model training, evaluation

## 📈 Model Performance (on held-out test set)

| Model              | Accuracy | F1-Score (weighted) |
|--------------------|----------|---------------------|
| SVM — LinearSVC    | 78.0%    | 78.1%               |
| Logistic Regression| 76.7%    | 76.8%               |
| Naive Bayes        | 64.1%    | 62.0%               |
| KNN                | ~51%     | ~51%                |
| Decision Tree      | ~47%     | ~40%                |

*Note: KNN and Decision Tree results depend on the best hyperparameter found by cross-validation.*

## 🚀 Future Work

* Use deep learning (LSTM, BERT)
* Real-time sentiment tracking
* Deployment as a web dashboard

## 👨‍💻 Author

Yuvraj, Anshil, Depanshu
