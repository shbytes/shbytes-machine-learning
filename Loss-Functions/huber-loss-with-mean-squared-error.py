import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Generate synthetic polynomial data (with some noise and outliers)
np.random.seed(42)

# Generate 100 samples (X)
X = np.sort(5 * np.random.rand(100, 1), axis=0)

# Generate polynomial target values (y = X^3 + Gaussian noise)
y = X**3 + np.random.randn(100, 1) * 10  # Cubic relationship with noise

# 2. Introduce some outliers
y[::10] = y[::10] + 50  # Every 10th value has a large outlier

# 3. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Transform the input data into polynomial features (degree 3 for cubic relationship)
poly = PolynomialFeatures(degree=3)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# 5. Initialize and train the Huber Regressor model (which uses Huber Loss)
model = HuberRegressor()
model.fit(X_poly_train, y_train)

# 6. Make predictions using the trained model
y_pred = model.predict(X_poly_test)

# 7. Calculate Mean Squared Error (MSE) for the predictions
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# 8. Visualize the results: Plot the true values, the predicted polynomial curve, and the outliers
X_range = np.linspace(0, 5, 1000).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_range_pred = model.predict(X_range_poly)

plt.scatter(X_test, y_test, color='blue', label='True values')
plt.plot(X_range, y_range_pred, color='red', label='Huber Regression Polynomial Fit')
plt.scatter(X[::10], y[::10], color='green', label='Outliers', s=100)  # Outliers
plt.xlabel("X")
plt.ylabel("y")
plt.title("Polynomial Regression with Huber Loss and MSE Evaluation")
plt.legend()
plt.show()

# Output => Mean Squared Error (MSE): 1041.9669985250653