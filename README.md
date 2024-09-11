# Titanic Classification

This project is a classification problem using the famous Titanic dataset. The objective is to predict whether a passenger survived or not based on features such as age, sex, ticket class, and others.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Modeling](#modeling)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Installation and Usage](#installation-and-usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Titanic Classification problem is one of the most well-known challenges for beginners in data science. The dataset contains demographic and travel information for 891 passengers, and the goal is to predict which passengers survived the Titanic disaster.

## Dataset

The dataset is provided by [Kaggle](https://www.kaggle.com/c/titanic). It contains the following columns:

- **PassengerId**: Unique ID for each passenger
- **Survived**: Survival (0 = No, 1 = Yes)
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Name**: Name of the passenger
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **SibSp**: Number of siblings/spouses aboard the Titanic
- **Parch**: Number of parents/children aboard the Titanic
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

You can download the dataset [here](https://www.kaggle.com/c/titanic/data).

## Modeling

The following steps were followed for building the model:

1. **Data Preprocessing**: 
    - Handling missing values (e.g., Age and Cabin)
    - Encoding categorical variables (e.g., Sex and Embarked)
    - Feature engineering (e.g., creating new features from existing data)
  
2. **Feature Selection**:
    - Selected relevant features such as `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, and `Fare`.

3. **Modeling**:
    - Implemented a classification model using the **Random Forest Classifier**. Other models such as **Logistic Regression** and **K-Nearest Neighbors** were also explored.

4. **Hyperparameter Tuning**:
    - Used **GridSearchCV** to tune the hyperparameters of the Random Forest model for better performance.

## Evaluation Metrics

The model was evaluated using several classification metrics:
- **Accuracy**: The percentage of correct predictions.
- **Precision**: The percentage of relevant results among the retrieved instances.
- **Recall**: The percentage of relevant instances that were correctly retrieved.
- **F1-Score**: The harmonic mean of precision and recall.

## Results

The best model achieved the following performance metrics:

- **Accuracy**: 83%
- **Precision**: 0.79
- **Recall**: 0.74
- **F1-Score**: 0.76

The Random Forest Classifier outperformed other models such as Logistic Regression and K-Nearest Neighbors in terms of both accuracy and F1-Score.

## Installation and Usage

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- Jupyter Notebook or any Python IDE
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

### Installation

1. Clone this repository:

```bash
git clone https://github.com/Harshit-kandoi/titanic.git
```

2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Run the Jupyter Notebook or Python script:

```bash
jupyter notebook Titanic.ipynb
```

### Usage

1. Load the Titanic dataset into the notebook.
2. Run each cell in the notebook to preprocess the data, train the model, and evaluate the results.
3. Modify the code to try different models and improve the results.

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
