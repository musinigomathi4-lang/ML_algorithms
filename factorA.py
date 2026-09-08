import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis

# Load Iris dataset
iris = load_iris()

# Create DataFrame
X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

print("Original Data:")
print(X.head())

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create Factor Analysis model
fa = FactorAnalysis(
    n_components=2,
    rotation='varimax', #interpretation
    random_state=42
)

# Fit and transform the data
X_factors = fa.fit_transform(X_scaled)

# Display factor loadings
loadings = pd.DataFrame(
    fa.components_.T,
    index=X.columns,
    columns=['Flower size', 'Sepal charac']
)

print("\nFactor Loadings:")
print(loadings)

# Display factor scores
factor_scores = pd.DataFrame(
    X_factors,
    columns=['Flower size', 'Sepal charac']
)

print("\nFactor Scores:")
print(factor_scores.head())

# Plot the factor scores
plt.figure(figsize=(8, 5))

plt.scatter(
    X_factors[:, 0],
    X_factors[:, 1],
    c=iris.target,
    cmap='viridis'
)

plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.title("Factor Analysis - Iris Dataset")
plt.colorbar(label="Species")

plt.show()
