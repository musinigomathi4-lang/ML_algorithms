import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("insurance.csv")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns)

X = df[["smoker", "age", "bmi", "children"]]
y = df["charges"]

X["smoker"] = X["smoker"].map({
    "no": 0,
    "yes": 1
})

print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Lasso(alpha=1.0)
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mse)
print("RMSE:", rmse)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print(results)