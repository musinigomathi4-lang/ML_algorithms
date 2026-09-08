import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# 1. Load the dataset
df = pd.read_csv("train.csv")

print("First 5 rows:")
print(df.head())


# 2. Select features for clustering
data = df[["Age", "Fare", "Pclass"]].copy()


# 3. Check missing values
print("\nMissing values:")
print(data.isnull().sum())


# 4. Fill missing Age values
data["Age"] = data["Age"].fillna(data["Age"].median())


# 5. Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(data)


# 6. Create K-Means model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)


# 7. Train the model and create clusters
clusters = kmeans.fit_predict(X)


# 8. Add cluster numbers to the dataset
data["Cluster"] = clusters


# 9. Display the clustered data
print("\nClustered Data:")
print(data.head(10))


# 10. Count customers/passengers in each cluster
print("\nNumber of passengers in each cluster:")
print(data["Cluster"].value_counts())


# 11. Plot the clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    data["Age"],
    data["Fare"],
    c=data["Cluster"],
    cmap="viridis"
)

plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Titanic Passengers - K-Means Clustering")
plt.colorbar(label="Cluster")

plt.show()