# StrokePredictionUsingMachinelearning-flask
A machine learning Flask app to predict stroke risk with explainability
# 🧠 Stroke Prediction System using Machine Learning and Flask

This project is a real-time web application that predicts the risk of a stroke based on clinical and lifestyle inputs. Built using a trained Logistic Regression model with SMOTE for class balancing and deployed using a Flask web framework, this system also incorporates explainable AI to show users why the prediction was made.

---

## 🚀 Features

- ✅ Predict stroke risk using Logistic Regression with SMOTE
- ✅ Flask-based interactive web interface
- ✅ Feature importance explanation using model coefficients
- ✅ Educational awareness (FAST symptoms)
- ✅ Background video, alert popups, and video for "Likely" prediction
- ✅ Frontend validation and responsive UI (Bootstrap)
- ✅ VotingClassifier ensemble model available in training script

---

## 🧪 Machine Learning Pipeline

- Preprocessing with `ColumnTransformer`
  - `OneHotEncoder` for categorical variables
  - `OrdinalEncoder` for smoking status
- Class imbalance handled using **SMOTE**
- Trained using multiple models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
- Final model: Logistic Regression with SMOTE (balanced and explainable)
