# 🌸 Iris Flower Classification

### OASIS INFOBYTE — Data Science Internship
### Data Science — Task 1

---

## 📌 Project Overview

This project develops a supervised machine learning classification system
for identifying the species of an Iris flower from its physical
measurements.

The project uses the classic Iris dataset provided through Scikit-learn.
Each observation contains four numerical measurements:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The target variable represents one of three Iris species:

- Iris Setosa
- Iris Versicolor
- Iris Virginica

The project follows a complete machine learning workflow, including data
inspection, exploratory data analysis, visualization, feature analysis,
model development, evaluation, comparison, and final model selection.

---

## 🎯 Objectives

The main objectives of this project are:

1. Load the Iris dataset using Scikit-learn.
2. Understand the structure and characteristics of the dataset.
3. Perform exploratory data analysis.
4. Check data types, missing values, duplicates, and descriptive statistics.
5. Analyze feature distributions.
6. Visualize relationships between features using a pairplot.
7. Compare feature distributions using box plots.
8. Analyze correlations between numerical features.
9. Discuss the importance of individual features.
10. Split the dataset into training and testing sets.
11. Train multiple classification algorithms.
12. Evaluate the models using multiple classification metrics.
13. Compare model performance.
14. Select the best-performing classifier.

---

## 🗂️ Dataset

The Iris dataset is loaded directly from Scikit-learn using:

```python
from sklearn.datasets import load_iris

iris = load_iris()