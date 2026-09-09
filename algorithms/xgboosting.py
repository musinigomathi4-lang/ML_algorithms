import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier


# Load dataset
df = pd.read_csv("breast-cancer.csv")


# Input features and target
X = df.drop(["diagnosis", "id"], axis=1)
y = df["diagnosis"]


# Convert B/M into 0/1
y = y.map({"B": 0, "M": 1})  #converted into numerical bc xgboost works with the numbers


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create XGBoost model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
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