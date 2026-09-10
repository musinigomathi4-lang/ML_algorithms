import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
df = pd.read_csv("breast-cancer.csv")

print(df.head())
print(df.info())

#select input and output features
X = df.drop(["diagnosis", "id"], axis=1)
y = df["diagnosis"]

#Dataset splitting
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#Adaboost model
model = AdaBoostClassifier(  #does not use max_depth directly as it uses a base estimator/ decision stump
    n_estimators=50,  #weak leaners are decision trees by default
    learning_rate=1.0, #contribution of each tree
    random_state=42
)
model.fit(X_train, y_train)

#Predictions
y_pred = model.predict(X_test)

# Training predictions
y_train_pred = model.predict(X_train)

# Training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)

#Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Training accuracy:", train_accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))