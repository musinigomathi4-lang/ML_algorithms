import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset
df = pd.read_csv("Iris.csv")

# Select features
X = df[['SepalLengthCm', 'SepalWidthCm']]

# Create linkage matrix
Z = linkage(X, method='ward')

# Create dendrogram
plt.figure(figsize=(10, 6))

dendrogram(Z)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")

plt.show()