# 🌸 Iris Flower Classification

### OASIS INFOBYTE — Data Science Internship
### Data Science — Task 1

---

## 📌 Project Overview

This project is part of the **OASIS INFOBYTE Data Science Internship** and focuses on developing a supervised machine learning classification system for identifying the species of an Iris flower from its physical measurements.

The project uses the well-known **Iris dataset**, which contains measurements of Iris flowers belonging to three different species:

- Iris Setosa
- Iris Versicolor
- Iris Virginica

The classification system uses four physical measurements of each flower:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The project follows a complete end-to-end machine learning workflow, beginning with dataset loading and inspection, followed by exploratory data analysis, visualization, feature analysis, data preparation, model training, evaluation, comparison, and final model selection.

---

# 🎯 Objectives

The main objectives of this project are:

1. Load the Iris dataset using Scikit-learn.
2. Understand the structure and characteristics of the dataset.
3. Inspect dataset dimensions and data types.
4. Check for missing values and duplicate records.
5. Generate descriptive statistics.
6. Analyze the distribution of the target classes.
7. Perform exploratory data analysis.
8. Visualize relationships between numerical features.
9. Create a pairplot/scatter matrix to study feature relationships.
10. Create box plots to compare feature distributions across species.
11. Analyze correlations between numerical features.
12. Discuss feature importance and feature selection.
13. Split the dataset into training and testing sets using an 80:20 ratio.
14. Apply appropriate feature scaling.
15. Train multiple classification algorithms.
16. Evaluate the models using accuracy, precision, recall, and F1-score.
17. Analyze confusion matrices.
18. Compare model performance.
19. Select the best-performing classification model.
20. Document the complete workflow in a reproducible Jupyter Notebook.

---

# 🗂️ Dataset

The Iris dataset is loaded directly from **Scikit-learn** using its built-in dataset loader.

```python
from sklearn.datasets import load_iris

iris = load_iris()
```

No external dataset download is required.

## Dataset Characteristics

| Property | Details |
|---|---|
| Dataset | Iris |
| Total observations | 150 |
| Input features | 4 |
| Target classes | 3 |
| Feature type | Numerical |
| Missing values | None |
| Classification type | Multiclass |
| Source | Scikit-learn built-in dataset |

---

# 📊 Features

The dataset contains four numerical input variables.

| Feature | Description |
|---|---|
| Sepal Length | Length of the flower sepal measured in centimeters |
| Sepal Width | Width of the flower sepal measured in centimeters |
| Petal Length | Length of the flower petal measured in centimeters |
| Petal Width | Width of the flower petal measured in centimeters |

The target variable represents the species of the Iris flower.

---

# 🌺 Target Classes

The classification problem contains three classes:

| Numerical Label | Species |
|---:|---|
| 0 | Iris Setosa |
| 1 | Iris Versicolor |
| 2 | Iris Virginica |

The numerical target values are converted into readable species names during the exploratory analysis.

---

# 🔬 Project Methodology

The project follows a structured machine learning pipeline:

```text
                Iris Dataset
                     │
                     ▼
             Dataset Inspection
                     │
                     ▼
        Exploratory Data Analysis
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   Distributions   Pairplot    Box Plots
        │            │            │
        └────────────┼────────────┘
                     ▼
             Correlation Analysis
                     │
                     ▼
              Feature Analysis
                     │
                     ▼
              Train/Test Split
                     │
                     ▼
              Feature Scaling
                     │
              ┌──────┴──────┐
              ▼             ▼
       Logistic Regression  KNN
              │             │
              └──────┬──────┘
                     ▼
              Model Evaluation
                     │
                     ▼
              Model Comparison
                     │
                     ▼
                Best Model
```

---

# 🔎 Exploratory Data Analysis

Exploratory Data Analysis was performed before model training to understand the structure, distribution, and relationships within the dataset.

## 1. Dataset Inspection

The following characteristics were examined:

- Dataset shape
- Column names
- Data types
- Missing values
- Duplicate records
- Descriptive statistics
- Species distribution

This initial inspection confirms that the dataset is clean and suitable for machine learning without requiring extensive data-cleaning operations.

---

## 2. Feature Distribution Analysis

Distribution plots were created for all four numerical features.

The analysis helps identify:

- Range of feature values
- Distribution patterns
- Differences between species
- Potential overlap between classes

### Key Observation

Petal-related measurements show clearer differences between the Iris species than many of the sepal measurements.

This indicates that **Petal Length** and **Petal Width** are likely to be particularly useful features for classification.

---

## 3. Pairplot / Scatter Matrix

A pairplot was created to visualize pairwise relationships between the numerical features.

The visualization allows us to examine:

- Relationships between features
- Species-wise clustering
- Separation between classes
- Potentially informative features

### Key Observation

The pairplot shows strong separation between **Iris Setosa** and the other two species.

Petal Length and Petal Width provide particularly strong separation between the three classes.

Versicolor and Virginica show some overlap, meaning that these two species can be relatively more difficult to distinguish based on certain measurements.

---

## 4. Box Plot Analysis

Box plots were generated for each numerical feature across the three species.

These plots allow comparison of:

- Median values
- Data spread
- Species-level differences
- Potential outliers

### Key Observation

The distributions of Petal Length and Petal Width differ substantially between the species.

This supports the observations obtained from the pairplot and indicates that petal measurements contain highly useful information for classification.

---

## 5. Correlation Analysis

A correlation matrix and heatmap were used to understand relationships between numerical features.

The analysis shows strong positive relationships between several measurements, particularly between the petal-related features.

This indicates that some variables contain related information while still providing useful signals for distinguishing the target classes.

---

# 🧠 Feature Selection Discussion

The exploratory analysis suggests that all four numerical features contain useful information.

However, **Petal Length** and **Petal Width** appear to be the most informative features because they provide strong visual separation between the Iris species.

**Sepal Length** and **Sepal Width** also contribute information, although there is greater overlap between the species for these measurements.

For the machine learning models, all four features were retained.

Keeping all four variables allows the models to use the complete information available in the dataset while avoiding premature feature elimination.

---

# ⚙️ Data Preparation

The dataset was divided into input features and the target variable.

## Input Features

```text
Sepal Length
Sepal Width
Petal Length
Petal Width
```

## Target

```text
Species
```

The dataset was divided using an **80:20 train-test split**.

```text
Training data: 80%
Testing data: 20%
```

A fixed random state was used to make the experiment reproducible.

Stratification was applied to preserve the class distribution in both the training and testing datasets.

---

# 📏 Feature Scaling

Feature scaling was performed using:

```python
StandardScaler()
```

Standardization transforms the features to approximately:

```text
Mean = 0
Standard Deviation = 1
```

The scaler was fitted only on the training data and then applied to both training and testing data.

This prevents information from the testing set from influencing the training process.

Feature scaling is especially important for the **K-Nearest Neighbors** algorithm because KNN relies on distances between observations.

---

# 🤖 Machine Learning Models

Two classification algorithms were implemented and compared.

---

## 1. Logistic Regression

Logistic Regression was used as one of the baseline classification models.

Although the name contains "Regression", Logistic Regression is commonly used for classification problems.

It estimates the probability that an observation belongs to each target class and assigns the most likely class to the observation.

### Advantages

- Simple and interpretable
- Computationally efficient
- Effective for many classification problems
- Provides a strong baseline model

---

## 2. K-Nearest Neighbors

K-Nearest Neighbors (KNN) is a distance-based classification algorithm.

For a new observation, KNN identifies the nearest observations in the training dataset and determines the predicted class based on their labels.

In this project, the number of neighbors was set to:

```text
K = 5
```

### Advantages

- Simple concept
- Effective for smaller datasets
- Naturally handles multiclass classification
- Can model nonlinear decision boundaries

Because KNN is distance-based, feature scaling was applied before training.

---

# 📈 Model Evaluation

Both models were evaluated on the unseen test dataset.

The following evaluation metrics were used.

## Accuracy

Accuracy measures the proportion of correctly classified observations.

```text
Accuracy =
Correct Predictions / Total Predictions
```

---

## Precision

Precision measures how many observations predicted as a particular class actually belong to that class.

High precision means that the model produces relatively few false-positive predictions.

---

## Recall

Recall measures how many observations belonging to a particular class were correctly identified.

High recall means that the model successfully identifies most observations belonging to that class.

---

## F1-Score

F1-score provides a balance between precision and recall.

It is useful when evaluating both types of classification errors rather than relying only on accuracy.

---

## Confusion Matrix

A confusion matrix provides a class-by-class breakdown of predictions.

It shows:

- Correct predictions
- Incorrect predictions
- Which classes are confused with one another

Separate confusion matrices were generated for both classification models.

---

# 🏆 Model Comparison

The project compares the two models using:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-Score

The notebook generates a detailed model comparison table.

The exact performance values are generated directly by the notebook using the specified train-test split and model configuration.

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | Generated by notebook | Generated by notebook | Generated by notebook | Generated by notebook |
| K-Nearest Neighbors | Generated by notebook | Generated by notebook | Generated by notebook | Generated by notebook |

The final model is selected based on its overall test-set performance while considering all evaluation metrics and the confusion matrices.

---

# 📊 Visualizations

The notebook contains the following visualizations.

## Exploratory Analysis

1. Feature distribution plots
2. Pairplot / scatter matrix
3. Species-wise box plots
4. Feature correlation heatmap

## Model Evaluation

5. Logistic Regression confusion matrix
6. KNN confusion matrix
7. Model accuracy comparison

These visualizations provide both statistical and visual perspectives on the classification problem.

---

# 💡 Key Insights

## 1. Iris Setosa is highly distinguishable

Setosa forms a clearly separated group in several feature relationships.

## 2. Petal measurements are highly informative

Petal Length and Petal Width show strong differences between the three species.

## 3. Versicolor and Virginica have some overlap

These species are less clearly separated than Setosa in certain feature combinations.

## 4. Multiple models perform strongly

Both Logistic Regression and KNN are capable of achieving strong classification performance on this dataset.

## 5. Model evaluation should use multiple metrics

Accuracy alone does not provide the complete picture, so precision, recall, F1-score, and confusion matrices are also considered.

---

# 🧪 Reproducibility

To make the experiment reproducible:

- A fixed `random_state=42` is used for the train-test split.
- Stratification is applied during splitting.
- The feature scaler is fitted only on the training data.
- The same transformation is applied to the test data.
- The model configuration is explicitly defined in the notebook.

This ensures that the workflow can be repeated consistently.

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Numerical operations |
| Pandas | Data manipulation and analysis |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning and evaluation |
| Jupyter Notebook | Interactive development and documentation |

---

# 📁 Project Structure

```text
DataScience-Task1-IrisClassification/
│
├── Iris_Flower_Classification.ipynb
├── README.md
├── requirements.txt
│
└── screenshots/
    ├── 01_dataset_overview.png
    ├── 02_feature_distributions.png
    ├── 03_pairplot.png
    ├── 04_boxplots.png
    ├── 05_correlation_heatmap.png
    ├── 06_confusion_matrices.png
    └── 07_model_comparison.png
```

---

# 🚀 How to Run the Project

## Prerequisites

Make sure Python 3.x is installed.

---

## 1. Clone the Repository

```bash
git clone https://github.com/atharva123-buddy/OIBSIP.git
```

Navigate to the project:

```bash
cd OIBSIP/DataScience-Task1-IrisClassification
```

---

## 2. Activate the Project Environment

The main internship repository uses a shared virtual environment at the OIBSIP root.

From the OIBSIP directory:

### Windows PowerShell

```powershell
.env\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```bash
pip install -r DataScience-Task1-IrisClassification/requirements.txt
```

---

## 4. Launch Jupyter Notebook

From the OIBSIP directory:

```bash
jupyter notebook
```

Open:

```text
DataScience-Task1-IrisClassification/Iris_Flower_Classification.ipynb
```

Run the notebook cells from top to bottom.

---

# 📸 Screenshots

Important project outputs should be stored inside the `screenshots/` directory.

Recommended screenshots include:

- Dataset inspection
- Feature distributions
- Pairplot
- Box plots
- Correlation heatmap
- Confusion matrices
- Model comparison

These screenshots provide visual evidence of the completed analysis and model evaluation.

---

# 📚 Learning Outcomes

This project provided practical experience with the complete machine learning workflow.

## Data Analysis

- Dataset inspection
- Data types
- Missing-value analysis
- Duplicate detection
- Descriptive statistics

## Exploratory Data Analysis

- Histograms
- Pairplots
- Scatter relationships
- Box plots
- Correlation heatmaps
- Feature interpretation

## Machine Learning

- Train/test splitting
- Feature scaling
- Supervised learning
- Multiclass classification
- Logistic Regression
- K-Nearest Neighbors

## Model Evaluation

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices
- Model comparison

## Documentation and Version Control

- Jupyter Notebook documentation
- Markdown documentation
- Requirements management
- Git version control
- GitHub repository management

---

# 🔮 Future Improvements

Although the current project satisfies the core classification requirements, several improvements could be explored:

- Test additional algorithms such as Decision Tree and Random Forest.
- Perform hyperparameter tuning.
- Use cross-validation for more robust evaluation.
- Experiment with automated feature-selection techniques.
- Build an interactive Iris species prediction application using Streamlit.
- Deploy the final model as a web application.
- Save the trained model using Joblib.
- Add automated testing and model validation.

---

# ✅ Project Status

**Status:** Completed

**Internship:** OASIS INFOBYTE Data Science Internship

**Task:** Task 1 — Iris Flower Classification

**Project Type:** Supervised Machine Learning — Multiclass Classification

---

# 👨‍💻 Author

**Atharva Joshi**

Data Science Intern  
OASIS INFOBYTE

---

# 📄 Disclaimer

This project was developed as part of the OASIS INFOBYTE Data Science Internship for educational and portfolio purposes.

The Iris dataset is used to demonstrate the machine learning classification workflow.
