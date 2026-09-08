import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
iris = load_iris()

# 2. Display the dataset
data = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

print(data)

# 3. Select independent variable (X) and dependent variable (y)
X = data[["petal length (cm)"]]
y = data["petal width (cm)"]

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 5. Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Get intercept and coefficient
print("Intercept:", model.intercept_) 
print("Coefficient:", model.coef_[0]) 

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Evaluate the model
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# 9. Display actual vs predicted values
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print(results)