# Bank Personal Loan Prediction and Customer Segmentation

A machine learning project using a retail banking dataset to predict whether a customer will accept a personal loan offer and to identify customer segments based on demographic, financial, and banking-behavior characteristics.

The project combines **K-Means clustering**, **K-Nearest Neighbors (K-NN)**, and an **Artificial Neural Network (ANN)**, followed by a comparison with an **Azure AutoML Voting Ensemble** model.

## Small Project Description

**A machine learning solution for a retail bank that uses customer demographic, financial, and banking data to predict personal-loan acceptance and discover customer segments using K-Means, K-NN, ANN, and Azure AutoML.**

## Project Overview

The project addresses two related machine learning tasks:

1. **Customer segmentation** using unsupervised learning.
2. **Personal loan acceptance prediction** using supervised classification.

The target variable is `Personal Loan`:

```text
0 = Customer did not accept the personal loan
1 = Customer accepted the personal loan
```

The dataset contains **5,000 customer records**, **13 input features**, and one binary target variable.

The overall workflow is:

```text
Bank Customer Dataset
        |
        v
Data Preparation
        |
        +----------------------+
        |                      |
        v                      v
   K-Means Clustering     Classification
        |                 /             \
        v                v               v
 Customer Segments      K-NN             ANN
                         |                |
                         +-------+--------+
                                 |
                                 v
                         Model Evaluation
                                 |
                                 v
                         Azure AutoML
                                 |
                                 v
                         Voting Ensemble
```

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Azure Machine Learning AutoML
- K-Means
- K-Nearest Neighbors
- Artificial Neural Networks
- StandardScaler
- Elbow Method
- Silhouette Analysis
- Classification metrics

## Dataset

The project uses the following dataset:

```text
bank_personal_loan_data.csv
```

The dataset contains **5,000 customers** and includes demographic, financial, and banking relationship information.

### Main Features

#### Demographic Features

- `Age`
- `Experience`
- `Family`
- `Education`

#### Financial Features

- `Income`
- `CCAvg`
- `Mortgage`

#### Banking Relationship Features

- `Securities Account`
- `CD Account`
- `Online`
- `CreditCard`

Other dataset fields include customer identifiers and location information.

The `ID` and `ZIP Code` fields are removed before modeling because they do not provide useful predictive information for the clustering and classification tasks.

## Machine Learning Tasks

### 1. K-Means Customer Segmentation

K-Means is used as an unsupervised learning technique to discover groups of customers with similar characteristics.

The target variable `Personal Loan` is not used during clustering.

Before clustering:

- `ID` is removed.
- `ZIP Code` is removed.
- `Personal Loan` is excluded.
- Numerical features are standardized using `StandardScaler`.

### Selecting K

Two techniques are used to determine the appropriate number of clusters:

- Elbow Method
- Silhouette Analysis

The analysis selected:

```text
K = 4
```

The four clusters represent different customer profiles based on characteristics such as income, credit-card spending, mortgage value, age, and banking-product usage.

The reported silhouette score for K = 4 was approximately:

```text
0.165
```

The clustering analysis is exploratory and is used to better understand customer behavior before classification.

## 2. K-Nearest Neighbors

K-NN is used as one of the supervised classification models to predict whether a customer will accept a personal loan.

### Configuration

```text
K = 5
```

The dataset is split into:

```text
80% Training
20% Testing
```

The input variables are standardized using `StandardScaler`.

### Reported Results

The K-NN model achieved:

```text
Accuracy: 96.0%
Precision: 94%
Recall: 62%
F1-Score: 75%
```

The classification metrics show that the model achieved high overall accuracy while still missing some customers who actually accepted the loan.

## 3. Artificial Neural Network

An ANN is used as a second supervised classification model.

### Architecture

```text
Input Layer
     |
     v
16 Neurons
     |
     v
8 Neurons
     |
     v
Output Layer
(Sigmoid)
```

The ANN uses:

- Adam optimizer
- Binary Cross-Entropy loss
- 20 training epochs
- Sigmoid output for binary classification

### Reported Results

The ANN achieved:

```text
Accuracy: 97.7%
Precision: 88%
Recall: 89%
F1-Score: 88%
```

These results were used as the baseline for comparison with the enhanced Azure AutoML model.

## 4. Azure AutoML

Azure AutoML is used to enhance the baseline solution through automated machine learning processes.

The Azure workflow includes:

- Automated feature engineering
- Data preprocessing
- Hyperparameter optimization
- Model selection
- 5-fold cross-validation
- Ensemble learning

The final Azure solution uses a **Voting Ensemble** model.

### Reported Results

| Metric | Baseline ANN | Azure Voting Ensemble |
|---|---:|---:|
| Accuracy | 97.70% | 98.76% |
| Precision | 88.00% | 98.75% |
| Recall | 89.00% | 98.76% |
| F1-Score | 88.00% | 98.74% |
| Weighted AUC | Not evaluated | 0.99808 |

These values are the results reported in the project analysis.

## Model Evaluation

The project evaluates classification models using multiple metrics rather than relying only on accuracy.

### Accuracy

Measures the proportion of correctly classified customers.

### Precision

Measures how many customers predicted as positive actually accepted the loan.

### Recall

Measures how many customers who actually accepted the loan were correctly identified.

### F1-Score

Combines precision and recall into a single metric.

### AUC

The Azure model also reports a weighted AUC to measure its ability to distinguish between the two classes.

## Data Preparation

Data preparation includes:

1. Removing unnecessary identifier fields.
2. Separating the target variable.
3. Splitting the data into training and testing sets.
4. Standardizing numerical features using `StandardScaler`.
5. Preparing the data for K-Means, K-NN, and ANN models.

Standardization is particularly important because K-Means and K-NN rely on distances between observations.

## Technical Decisions

Several technical decisions were made during the project:

- Removing `ID` and `ZIP Code` to reduce irrelevant information.
- Excluding `Personal Loan` from K-Means to prevent target leakage into clustering.
- Standardizing input features before distance-based models.
- Using Elbow and Silhouette methods to select K.
- Using K-NN as an interpretable classification baseline.
- Using ANN to model more complex relationships.
- Using Azure AutoML to automate model selection and optimization.

## How to Run

### Requirements

Install Python and the main libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
```

Azure AutoML components require an Azure Machine Learning environment and appropriate Azure credentials/resources.

### Run the Python Notebook

1. Clone or download the repository.
2. Make sure the dataset is in the same directory as the notebook.
3. Confirm the dataset filename:

```text
bank_personal_loan_data.csv
```

4. Start Jupyter Notebook:

```bash
jupyter notebook
```

5. Open:

```text
Yousof_Flaifel_AI.ipynb
```

6. Run the notebook cells in order.

### Expected Workflow

The notebook should follow the general process:

```text
Load Dataset
     |
     v
Explore Data
     |
     v
Prepare Features
     |
     v
Standardize Data
     |
     +-------------------+
     |                   |
     v                   v
 K-Means              Classification
     |                   |
     v                   +--------+
Customer Segments       |        |
                        v        v
                       K-NN     ANN
                        |        |
                        +----+---+
                             |
                             v
                       Evaluate Models
```

## Project Files

The main files for the project are:

```text
Yousof_Flaifel_AI.ipynb
Yousof_Flaifel_AI.docx
bank_personal_loan_data.csv
Dataset Metadata.pdf
```

### `Yousof_Flaifel_AI.ipynb`

Contains the Python-based machine learning implementation and analysis.

### `Yousof_Flaifel_AI.docx`

Contains the project report, methodology, model discussion, evaluation, technical decisions, assumptions, challenges, and future improvements.

### `bank_personal_loan_data.csv`

Contains the customer data used for the machine learning tasks.

### `Dataset Metadata.pdf`

Contains information describing the dataset and its variables.

## Limitations

The project makes several assumptions:

- Historical customer behavior is representative of future behavior.
- The `Personal Loan` target is correctly labelled.
- The available features contain enough information to predict loan acceptance.
- Future customer behavior will follow patterns similar to the historical dataset.

The clustering analysis also has limitations. The reported silhouette score indicates that the customer groups are not strongly separated, so the clusters should be interpreted as exploratory customer segments rather than definitive categories.

## Future Improvements

Possible improvements include:

- Adding more customer records.
- Adding transaction history.
- Adding previous loan history.
- Adding customer engagement data.
- Performing additional hyperparameter tuning.
- Testing more advanced ensemble models.
- Adding Explainable AI techniques.
- Further validating the models on new and external data.

## Learning Outcomes

This project demonstrates practical experience with:

- Supervised machine learning
- Unsupervised machine learning
- Customer segmentation
- Binary classification
- K-Means clustering
- K-Nearest Neighbors
- Artificial Neural Networks
- Data preprocessing
- Feature selection
- Feature scaling
- Model evaluation
- Accuracy, precision, recall, F1-score, and AUC
- Elbow Method
- Silhouette Analysis
- TensorFlow / Keras
- Scikit-learn
- Azure AutoML
- Ensemble learning

## Author

**Yousof Flaifel**

Data Science & AI Student  
Al Hussein Technical University (HTU)

---

University Artificial Intelligence and Machine Learning Project
