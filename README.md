# Bank Credit Risk Assessment

A machine learning web application that evaluates the credit risk of loan applicants using the German Credit Dataset. Given an applicant's financial and personal details, the model predicts whether they are a good or bad credit risk and provides a probability score.  
Created as a final project for our Python Pandas Practice in YSCI  
Live demo: https://bankriskmanagment.streamlit.app

---

## Overview

Banks and financial institutions need reliable tools to assess whether a loan applicant is likely to repay their debt. This project trains a classification model on historical credit data and exposes it through an interactive web interface where analysts can enter applicant details and receive an instant risk verdict.

---

## Dataset

**German Credit Dataset**  
Source: https://www.kaggle.com/datasets/mpwolke/cusersmarildownloadsgermancsv

- 1000 applicant records
- 20 feature columns covering financial status, personal profile, and assets
- Binary target: `Creditability` (1 = Good risk, 0 = Bad risk)
- Class distribution: 700 good / 300 bad

---

## Model

The application uses a Logistic Regression classifier trained with balanced class weights to account for the class imbalance in the dataset.

Preprocessing steps:
- Standard scaling applied to `Credit_Amount`, `Age_years`, and `Duration_of_Credit_monthly`
- All other features are ordinal encoded as per the original dataset specification

Model artifacts saved with `joblib`:
- `credit_model.pkl` — trained classifier
- `scaler.pkl` — fitted StandardScaler

Performance on the test set (200 samples, 80/20 split):

| Metric    | Score |
|-----------|-------|
| Accuracy  | ~0.74 |
| Precision | ~0.83 |
| Recall    | ~0.74 |
| ROC-AUC   | ~0.79 |

---

## Features

- 3-column input form covering loan details, personal profile, and assets
- Real-time prediction with good/bad risk verdict
- Probability score displayed as a percentage
- Bar chart and donut chart visualizing the risk breakdown
- Risk level classification: Low / Medium / High
- Contextual insight text based on the prediction

---

## Project Structure

```
bankriskmanagment/
├── data/
│   └── german.csv             # Raw dataset
├── app.py                     # Streamlit web application
├── sample.ipynb               # Data exploration, model training, and evaluation
├── credit_model.pkl           # Trained Logistic Regression model
├── scaler.pkl                 # Fitted StandardScaler
├── columns.pkl                # Saved column order used during training
├── requirements.txt           # Python dependencies
├── LICENSE
└── README.md
```
---

## Installation

**Prerequisites:** Python 3.9 or higher

1. Clone the repository:
   ```bash
   git clone https://github.com/BabkenInants/BankRiskManagment.git
   cd bankriskmanagment
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser at `http://localhost:8501`

---

## Dependencies

```
streamlit
pandas
scikit-learn
joblib
matplotlib
numpy
```

---

## How to Use

1. Fill in the three sections of the form: Loan Details, Personal Profile, and Assets & Housing
2. Click "Evaluate Credit Risk"
3. The result card shows the verdict, probability score, risk level, and a short insight
4. The charts on the right break down the good vs bad risk probabilities visually

---

## Authors

Khachatur Khojoyan  
[Github](https://github.com/KhachaturKhojoyan)  
[Linkedin](https://www.linkedin.com/in/khachatur-khojoyan-1327343b0/)  
  
Babken Inants  
[Github](https://www.github.com/BabkenInants)  
[Linkedin](https://www.linkedin.com/in/inantsbabken/)  

---

## License
[MIT License](/LICENSE)

This project was built for educational purposes as part of a machine learning practice assignment. The dataset we used is publicly available on Kaggle under its original license.
