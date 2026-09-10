import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.ensemble import StackingClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# Load dataset
df = pd.read_csv("breast-cancer.csv")


# Input features and target
X = df.drop(["diagnosis", "id"], axis=1)
y = df["diagnosis"]


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Base models
base_models = [
    ("logistic", make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )),

    ("knn", make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )),

    ("decision_tree", DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )),

    ("svm", make_pipeline(
        StandardScaler(),
        SVC(probability=True, random_state=42)
    ))
]


# Meta-model
meta_model = LogisticRegression(max_iter=1000)


# Create stacking model
model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5  #k fold cross validation
)


# Train
model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)


# Accuracy
training_accuracy = model.score(X_train, y_train)
testing_accuracy = accuracy_score(y_test, y_pred)

print("Training accuracy:", training_accuracy)
print("Testing accuracy:", testing_accuracy)


# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))