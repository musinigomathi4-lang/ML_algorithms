import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load dataset
df = pd.read_csv("breast-cancer.csv")

print(df.head())
print(df.info())


# 2. Select useful columns
'''df = df[
    [
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    ]
]


# 3. Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


# 4. Convert categorical columns into numbers
le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])'''


# 5. Separate input and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]


# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 7. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=12, #no of decision trees in that forest
    max_depth=3, #max depth for each tree
    random_state=42
)


# 8. Train model
model.fit(X_train, y_train)


# 9. Make predictions
y_pred = model.predict(X_test)


# Training predictions
y_train_pred = model.predict(X_train)

# Training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)

print("Training Accuracy:", train_accuracy)
# 10. Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


for i in range(5):

    plt.figure(figsize=(25, 12))

    plot_tree(
        model.estimators_[i],
        feature_names=X.columns,
        class_names=["Has tumor", "Has no tumor"],
        filled=True,
        rounded=True,
        fontsize=8
    )

    plt.title(f"Random Forest - Decision Tree {i + 1}")
    plt.show()