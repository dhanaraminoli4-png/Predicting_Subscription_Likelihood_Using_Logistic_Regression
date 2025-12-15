# Predicting_Subscription_Likelihood_Using_Logistic_Regression

Project Overview

This project demonstrates how to use logistic regression to predict whether a user subscribes to a service based on several features such as Age, Hours Studied Per Week, Completed Free Courses, and Forum Visits.

The project also includes data preprocessing, feature scaling, handling missing values, and visualization using PCA to reduce dimensionality for easy interpretation.

Dataset
CSV file: logistic_regression_project.csv

Features:
Age – Age of the user
- Hours_Studied_Per_Week – Average study hours per week
- Completed_Free_Courses – Number of free courses completed
- Visited_Forum – Number of forum visits
- Education_Level – (categorical, optional)

Target:
Subscribed – Whether the user subscribed (0 = No, 1 = Yes)

#Key Steps in the Project

Data Exploration:
- Displayed first rows of the dataset, info, and descriptive statistics.
- Checked for missing values and duplicates.
- Visualized feature distributions with histograms and boxplots.
- Checked relationships with the target using boxplots.
- Computed correlation heatmaps to detect feature dependencies.

Data Preprocessing:
- Handled missing values with SimpleImputer.
- Normalized numerical features using StandardScaler.
- Removed duplicate rows to ensure clean data.

Dimensionality Reduction for Visualization:
- Applied PCA to reduce 4 numerical features to 2 principal components (PC1 and PC2).
- This allows visualization of the decision boundary in 2D.

Logistic Regression:
- Split the dataset into train and test sets randomly.
- Trained a logistic regression model on PCA components.
- Evaluated model performance using accuracy score.

Visualization:
- Plotted decision boundary of the logistic regression model in 2D using PCA components.
- Plotted feature distributions and relationship with the target.

Dependencies:

- The project uses the following Python libraries:

pandas
numpy
matplotlib
seaborn

