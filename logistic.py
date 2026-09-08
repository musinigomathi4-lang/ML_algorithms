import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the dataset
df = pd.read_csv("train.csv")

print("First 5 rows:")
print(df.head())
print(df.columns)

# 2. Select the columns required for Logistic Regression
df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]

# 3. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# 4. Fill missing Age values with the median
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# 5. Convert Sex into numerical values
# female = 0
# male = 1
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])

# 6. Select input variables
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]

# 7. Select target variable
y = df["Survived"]

# 8. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 9. Standardize the data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 10. Create Logistic Regression model
model = LogisticRegression()


# 11. Train the model
model.fit(X_train, y_train)


# 12. Make predictions
y_pred = model.predict(X_test)


# 13. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)


# 14. Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 15. Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 16. Predict a new passenger
# Pclass = 3
# Sex = 1 (male)
# Age = 25
# SibSp = 0
# Parch = 0
# Fare = 10

new_passenger = [[3, 1, 25, 0, 0, 10]]

new_passenger = scaler.transform(new_passenger)

prediction = model.predict(new_passenger)


# 17. Display prediction
if prediction[0] == 1:
    print("\nPrediction: Passenger survived")
else:
    print("\nPrediction: Passenger did not survive")
