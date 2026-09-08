import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Load dataset
df = pd.read_csv("Iris.csv")


# 2. Remove ID column
df = df.drop("Id", axis=1)


# 3. Separate features and target
X = df.drop("Species", axis=1)
y = df["Species"]


# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y #ensures that the class distribution remains the same
)


# 5. Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 6. Create KNN model
model = KNeighborsClassifier(n_neighbors=5)


# 7. Train
model.fit(X_train_scaled, y_train)


# 8. Predict
y_pred = model.predict(X_test_scaled)


# 9. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

for species in y.unique():
    subset = df[df["Species"] == species]

    plt.scatter(
        subset["PetalLengthCm"],
        subset["PetalWidthCm"],
        label=species
    )

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Iris Dataset")
plt.legend()
plt.show()