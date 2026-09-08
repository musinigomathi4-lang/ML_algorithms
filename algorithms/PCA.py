import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Iris.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

X = df.iloc[:,:-1]  #removing the species col

X_std = (X-X.mean())/ X.std()  #standardizing

print("co variance")
cov_matrix = np.cov(X_std.T)   #finding covariance matrix
print(cov_matrix)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\n eigenvalues", eigenvalues)
print("\n eigenvectors", eigenvectors)

sorted_index = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_index]
eigenvectors = eigenvectors[:,sorted_index]

plt.plot(range(1, len(eigenvalues)+1), eigenvalues, marker='o')
plt.xlabel("principle components")
plt.ylabel("eigenvalues")
plt.title("scree plot")
plt.grid()
plt.show()