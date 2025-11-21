# Question-Pair-Similarity-Classification

## Overview

This project tackles the AI Developer (Night Shift) technical assignment from SparkTech Agency. The goal is to classify pairs of questions from the Quora dataset as **duplicates** or **not duplicates** (semantically similar or not). The solution includes exploratory data analysis (EDA), text preprocessing, feature extraction, model training using an Artificial Neural Network (ANN), and model evaluation.

## Dataset

* **train.csv**: Contains labeled question pairs.

  * `id`: Unique identifier for each question pair
  * `qid1` / `qid2`: Unique identifiers for the two questions
  * `question1` / `question2`: Text of each question
  * `is_duplicate`: Target variable (1 if duplicate, 0 otherwise)
* **Test set**: Contains question pairs for prediction (if included in the notebook).

## Project Structure

```
QuestionPairSimilarity/
│
├── assignment.ipynb        # Google Colab notebook with full workflow
├── train.csv               # Dataset provided by SparkTech
└── README.md               # This file
```

## Steps Implemented

### 1. Exploratory Data Analysis (EDA)

* Visualized distribution of `is_duplicate`.
* Analyzed text columns (`question1` and `question2`) using:

  * Word clouds
  * Sentence and word length distributions
  * Common words and patterns
* Checked for missing values and outliers.
* Investigated correlations between features like question length and character count.

### 2. Text Preprocessing

* Tokenization: Split text into words.
* Lowercasing for standardization.
* Removed stopwords, punctuation, and special characters.
* Applied **lemmatization** to normalize words.
* Converted text to numerical features using **TF-IDF vectors**.

### 3. Model Building

* Built an **Artificial Neural Network (ANN)** for classification.
* Considered sequence models (LSTM/GRU) for capturing semantic similarity.
* Used a baseline model for comparison (optional: Logistic Regression/SVM).
* Compiled model using **Adam optimizer** and **binary cross-entropy loss**.

### 4. Model Evaluation

* Metrics used:

  * Accuracy
  * Precision, Recall, F1-score
  * Confusion Matrix
  * AUC-ROC Curve
* Evaluated model performance on validation/test split.

### 5. Model Tuning

* Adjusted architecture: number of layers, neurons, and activation functions.
* Tuned hyperparameters: learning rate, batch size, and number of epochs.

## How to Run

1. Open `assignment.ipynb` in **Google Colab** or **Jupyter Notebook**.
2. Ensure `train.csv` is in the same folder.
3. Run cells sequentially to reproduce all results, visualizations, and metrics.

## Key Decisions & Notes

* **Model Choice:** ANN selected for flexibility and good performance on text similarity tasks.
* **Text Representation:** TF-IDF used to capture word importance across questions.
* **Evaluation Metrics:** Accuracy and F1-score prioritized due to class imbalance.
* **Assumptions:** Dataset is representative of Quora question similarity; no external datasets used.

## Contact

For any questions regarding this project, please reach out to:
Mahin
