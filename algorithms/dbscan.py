import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Iris.csv")

# Select features
X = df[['SepalLengthCm', 'SepalWidthCm']]

# Standardize the data
X_scaled = StandardScaler().fit_transform(X)

# Create DBSCAN model
model = DBSCAN(eps=0.5, min_samples=5)

# Create clusters
labels = model.fit_predict(X_scaled)

# Print cluster labels
print("Cluster labels:")
print(labels)

# Plot clusters
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    cmap='viridis'
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("DBSCAN Clustering")
plt.show()