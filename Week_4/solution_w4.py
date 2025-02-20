import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris

# Task 1: Data Preprocessing
# Load the dataset
data = pd.read_csv("heart.csv", sep=";")

print(data.head())

# Identify features and target variable
X = data.drop("cardio", axis=1)
y = data["cardio"]

# Identify missing values and fill them
missing_values = X.isnull().sum()
print(missing_values)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Task 2: Train Machine Learning models for classification
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)    
    print(f"{name}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.2f}\n")

# Task 3: Hyperparameter Tuning with GridSearchCV
from sklearn.model_selection import GridSearchCV

# param_grid = {
#     "C": [0.001, 0.01, 0.1, 1, 10, 100],
#     "kernel": ["linear", "poly", "rbf", "sigmoid"],
#     "gamma": ["scale", "auto"]
# }
# grid_search = GridSearchCV(SVC(), param_grid, cv=5)

# grid_search.fit(X_train, y_train)
# print(f"Best parameters: {grid_search.best_params_}")
# print(f"Best score: {grid_search.best_score_}")

param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"]
}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

# Task 4: Train Machine Learning models for regression
boston_data = pd.read_csv("boston.csv")

print(boston_data.head())

X_reg = boston_data.drop("medv", axis=1)
y_reg = boston_data["medv"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

models_reg = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
}

for name, model in models_reg.items():
    model.fit(X_train_reg, y_train_reg)
    y_pred_reg = model.predict(X_test_reg)
    print(f"{name}:")
    print(f"MSE: {mean_squared_error(y_test_reg, y_pred_reg):.2f}")
    print(f"R^2: {r2_score(y_test_reg, y_pred_reg):.2f}\n")

# Task 5: Clustering on Iris dataset
iris = load_iris()
X_iris = iris.data

clustering_models = {
    "KMeans": KMeans(n_clusters=3),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "Agglomerative Clustering": AgglomerativeClustering(n_clusters=3),
    "Gaussian Mixture Model": GaussianMixture(n_components=3)
}

for name, model in clustering_models.items():
    y_pred_cluster = model.fit_predict(X_iris)
    plt.scatter(X_iris[:, 0], X_iris[:, 1], c=y_pred_cluster, cmap='viridis', label=name)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(name)
    plt.show()
