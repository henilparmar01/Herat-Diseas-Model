# ❤️ Heart Disease Prediction using Machine Learning

## 📌 Project Overview

This project predicts whether a patient is likely to have heart disease based on clinical and medical attributes. Multiple machine learning classification algorithms are trained and compared to identify the best-performing model.

The project follows a complete machine learning pipeline, including data preprocessing, feature engineering, model training, evaluation, and performance comparison.


## 🎯 Objectives

- Predict the presence of heart disease.
- Compare multiple machine learning classification models.
- Handle missing values effectively.
- Perform feature preprocessing automatically.
- Evaluate models using multiple performance metrics.
- Select the best model based on ROC-AUC score.


## 📂 Dataset

**Dataset:** Heart Disease UCI Dataset

The dataset contains patient medical information such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- ECG Results
- Maximum Heart Rate
- Exercise-Induced Angina
- ST Depression
- ST Slope
- Number of Major Vessels
- Thalassemia

### Target Variable

- **0 → No Heart Disease**
- **1 → Heart Disease Present**



## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost



## 📚 Machine Learning Models

The following classification algorithms are implemented:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier
- XGBoost Classifier


## ⚙ Data Preprocessing

The project includes a complete preprocessing pipeline:

### Numerical Features

- Missing value imputation using Median
- Feature Scaling using StandardScaler

### Categorical Features

- Missing value imputation using Most Frequent value
- One-Hot Encoding

### Pipeline Components

- Pipeline
- ColumnTransformer
- SimpleImputer
- StandardScaler
- OneHotEncoder


## ✂ Train-Test Split

The dataset is divided into:

- Training Data: 80%
- Testing Data: 20%

The split uses:

- Random State = 42
- Stratified Sampling



## 📊 Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

The best model is selected based on the highest ROC-AUC score.


## 📈 Project Workflow

1. Load Dataset
2. Rename Columns
3. Create Target Variable
4. Remove Unnecessary Columns
5. Separate Features and Target
6. Identify Numerical and Categorical Columns
7. Build Preprocessing Pipeline
8. Split Dataset
9. Train Multiple Models
10. Evaluate Performance
11. Compare Results
12. Select Best Model


## 📁 Project Structure

```
Heart-Disease-Prediction/
│
├── heart_disease_uci.csv
├── heart_disease.py
├── README.md
└── requirements.txt
```


## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/henilparmar01/Heart-Disease-Prediction.git
```

Move into the project folder:

```bash
cd Heart-Disease-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python heart_disease.py
```

---

## 📦 Required Libraries

```
pandas
numpy
scikit-learn
xgboost
```

or install manually:

```bash
pip install pandas numpy scikit-learn xgboost
```


## 📌 Features

- Clean and organized code
- Automatic preprocessing pipeline
- Missing value handling
- Feature scaling
- Categorical encoding
- Multiple ML models
- Performance comparison
- ROC-AUC based model selection
- Easy to extend with additional algorithms



## 📊 Sample Output

The project displays:

- Numerical Columns
- Categorical Columns
- Model Comparison Table
- Best Performing Model

Example:

```
Model Comparison

Model                  Accuracy    Precision    Recall    F1 Score    ROC-AUC

Logistic Regression    0.842391    0.841121     0.882353  0.861244    0.903515
SVM                    0.842391    0.921569     0.866359  0.866359    0.916667
Random Forest          0.847826    0.849057     0.882353  0.865385    0.919955
XGBoost                0.858696    0.845455     0.911765  0.877358    0.903874

Best Model: Random Forest
```

*(Results may vary depending on dataset version and random seed.)*


## 🔮 Future Improvements

- Hyperparameter Tuning
- Cross Validation
- Feature Selection
- Model Saving using Joblib
- Flask/Django Web Application
- Streamlit Dashboard
- Explainable AI using SHAP
- Model Deployment


## 👨‍💻 Author @henilparmar01

**Henil Parmar**

Machine Learning Enthusiast

---

## ⭐ If you found this project helpful, don't forget to star the repository!# Herat-Diseas-Model
