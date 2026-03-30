# Ames Housing Price Prediction and Analytics

This repository contains a comprehensive Supervised Machine Learning (SML) project focused on predicting residential real estate prices in Ames, Iowa. The project is deployed as an interactive dashboard that allows for real-time model inference and performance benchmarking.

### Live Application
Access the interactive dashboard here: https://ames-housing-ml-project-caclvosnezmfffrrqo29oe.streamlit.app/

---

## Project Overview

The objective of this project is to analyze 81 distinct features of residential properties to provide accurate market valuations. The workflow covers the entire machine learning pipeline, including automated data preprocessing, feature engineering, comparative model training, and deployment.

### Key Sections

1. **Market Overview**: Provides high-level statistics of the dataset, including total listings, neighborhood-wise price distributions, and property value tiers.
2. **Feature Selection**: Utilizes a Random Forest Regressor to compute Gini Importance, identifying the top 10 variables that most significantly impact property value.
3. **Property Value Estimator**: A dynamic inference tool where users can input specific property details to receive a price prediction from the trained XGBoost model.
4. **Model Analytics**: A technical comparison of four distinct algorithms across regression and classification metrics.

---

## Machine Learning Architecture

### Models Evaluated
The application trains and benchmarks four different models:
* **Linear Regression**: Used as a baseline for linear feature relationships.
* **Support Vector Regression (SVR)**: Evaluated for its effectiveness in high-dimensional space.
* **Random Forest**: Utilized for robust non-linear modeling and feature importance extraction.
* **XGBoost (Gradient Boosting)**: Selected as the primary production model due to its superior performance in handling variance and minimizing Mean Absolute Error (MAE).

### Performance Metrics
Models are evaluated using the following criteria:
* **Regression**: Mean Absolute Error (MAE) and R-Squared (R2) score.
* **Classification**: Area Under the Curve (AUC), Precision, Recall, and F2-Score.
* **Selection Logic**: XGBoost was chosen for the final estimator as it maximized the F2-Score, ensuring the model remains sensitive to identifying high-value "Premium" listings.

---

## Data and Preprocessing

The project utilizes the Ames Housing dataset, which includes 2,930 observations. The preprocessing pipeline handles:
* **Missing Values**: Numerical data is imputed with medians; categorical data is handled via a 'None' classification.
* **Categorical Encoding**: Text-based features are transformed using Label Encoding to maintain compatibility with tree-based models.
* **Feature Scaling**: Standard scaling is applied specifically for the Linear Regression and SVR models.

---
