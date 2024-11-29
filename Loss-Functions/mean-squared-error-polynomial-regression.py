# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create a synthetic dataset that follows a parabolic relationship
# Generate X values (random data)
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)  # 80 samples between 0 and 5

# Create the parabolic target values (y = ax^2 + bx + c + noise)
y = 0.5 * X**2 - X + np.random.randn(80, 1) * 0.5  # Parabolic relationship with some noise

# 2. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize the PolynomialFeatures class with degree 2 for a parabolic fit
poly = PolynomialFeatures(degree=2)

# 4. Transform the input data to include polynomial features (X^2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 5. Train a linear regression model using the transformed polynomial features
model = LinearRegression()
model.fit(X_train_poly, y_train)

# 6. Make predictions on the test data
y_pred = model.predict(X_test_poly)

# 7. Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# 8. Visualize the results
# Create a fine grid of X values for plotting the parabolic curve
X_grid = np.linspace(0, 5, 100).reshape(-1, 1)
X_grid_poly = poly.transform(X_grid)
y_grid = model.predict(X_grid_poly)

# Plot the true values, predicted values, and the parabolic curve
plt.scatter(X_test, y_test, color='blue', label='True values')
plt.plot(X_grid, y_grid, color='red', label='Parabolic regression curve (predicted)')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Polynomial Regression (Parabolic) - True vs Predicted Values")
plt.legend()
plt.show()

# Output => Mean Squared Error (MSE): 0.1701467841288596