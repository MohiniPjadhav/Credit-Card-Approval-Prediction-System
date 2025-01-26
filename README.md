# Credit Card Approval Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![ML](https://img.shields.io/badge/Machine%20Learning-Classification-orange)

A machine learning system that predicts credit card approval likelihood using advanced feature engineering and imbalance-handling techniques. Built with production-ready pipelines and explainable AI components.


## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)

## Features

### Dataset Attributes
- **Demographic**: `person_age`, `person_income`, `person_home_ownership`
- **Financial**: `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`
- **Historical**: `cb_person_default_on_file`, `cb_person_cred_hist_length`
- **Target**: `loan_status` (0 = Approved/Non-Default, 1 = Default)

### Technical Highlights
- Advanced imbalance handling (ADASYN + class weights)
- Threshold optimization for business metrics
- Production-ready pipelines with ColumnTransformer
- Comprehensive visual diagnostics 

## Installation

1. Clone repository:
```bash
git clone https://github.com/MohiniPjadhav/Credit-Card-Approval-Prediction-System.git
cd credit-approval-prediction
```
2. Prediction Api
```bash
from flask import Flask, request
import joblib

app = Flask(__name__)
model = joblib.load('models/improved_credit_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    return {'risk_score': float(model.predict_proba([data])[0][1])}
