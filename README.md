# Machine Learning - Classification Project

This repository contains my project work for the Machine Learning 1 course at the University of Warsaw. The project focuses on predicting the willingness of individuals to change jobs using various machine learning techniques for binary classification.

## Overview

As part of this project, I utilized a dataset provided by the course, where the task was to predict whether someone is willing to change jobs based on several categorical and numerical features. The primary evaluation metric for model performance was balanced accuracy score. Additionally, the trained model was used to predict on an unlabeled dataset, where I achieved top performance in my class.

- **Balanced Accuracy on Validation Set:** 0.7580
- **Performance on Unlabeled Dataset:** Top performance in class

## Exploratory Data Analysis and Feature Engineering

## Dateset


The dataset provided by the university includes 12427 observations in the training sample and 3308 in the test sample and the following columns:

- id – unique observation identifier
- gender – gender of a person
- age – age of a person in years
- education – highest formal education level of a person attained so far
- field_of_studies – field of studies of a person
- is_studying – information whether a person is currently studying
- county – county code in which the person currently lives and works
- `relative_wage`` – relative wage in the county (as percentage of country average)
- years_since_job_change – years since a person last changed the job
- years_of_experience – total number of years of professional experience of a person
- hours_of_training – total number of training hours completed by a person
- is_certified – does a person have any formal certificate of completed trainings
- size_of_company – size of a company in which a person currently works
- typs_of_company – type of a company in which a person currently works
- transaction_amount_ratio – ratio in total amount of transactions in the 4th quarter against the 1st quarter
- willing_to_change_job – is a person willing to change job (outcome variable, only in the training sample)

I started with Exploratory Data Analysis (EDA) to understand the dataset better. Here are some highlights of the preprocessing steps and exploratory analysis:

- Cleaned and transformed features like `years_of_experience` and `county`.
- Binned `years_of_experience` into categories.
- Handled missing values and converted them into separate categories for categorical variables.
- Analyzed the distribution and relationship of categorical features with the target variable (`willing_to_change_job`).
- Converting categorical variables into numerical format using one-hot encoding.
- Transforming skewed numerical variables using the Yeo-Johnson transformation.

## Model Training and Evaluation

### Baseline Models

I initially evaluated several baseline models without hyperparameter tuning:

- Logistic Regression, SGDClassifier, Decision Tree, Random Forest, etc.
- **Best Performing Baseline Model:** Random Forest (Balanced Accuracy: 0.657)

### Model Optimization

For optimizing the Random Forest model, I used the following approach:

- Applied Yeo-Johnson transformation to normalize numerical features.
- Utilized SMOTE to handle class imbalance.
- Conducted hyperparameter tuning using Randomized Search with Stratified KFold cross-validation.

### Final Model Performance

The optimized Random Forest model achieved the following results on the test and validation sets:

- **Validation Set:**
  - Accuracy: 0.7812
  - Balanced Accuracy: 0.7580
  - Precision: 0.5459
  - Recall: 0.7120
  - F1 Score: 0.6180
  - ROC AUC Score: 0.7908

## Conclusion

This project provided valuable insights into the application of machine learning algorithms for classification tasks, focusing on predicting job change willingness. The optimization process, including feature engineering, model selection, and hyperparameter tuning, significantly improved model performance. For more details, please refer to the Jupyter notebook in this repository.


## Libraries Used:

The following Python libraries were utilized for this project:

- pandas - Data manipulation and analysis
- numpy - Mathematical functions on arrays
- seaborn - Data visualization
- matplotlib - Plotting library
- scikit-learn - Machine learning library for classification models
- imbalanced-learn - Library for handling class imbalance using techniques like SMOTE
- catboost, lightgbm, xgboost - Gradient boosting libraries for ensemble models

## Prerequisites
To run the code and reproduce the results, ensure you have the following installed:
- Python (version 3.7 or higher recommended)
- Jupyter Notebook (optional, for running Python code interactively)
- Required Python libraries: pandas, numpy, seaborn, matplotlib, scikit-learn, imbalanced-learn, catboost, lightgbm, xgboost

