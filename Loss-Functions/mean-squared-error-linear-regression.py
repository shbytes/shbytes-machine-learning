# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

# Create a synthetic dataset
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()  # Initialize the model

model.fit(X_train, y_train) # Train the model using the training data
y_pred = model.predict(X_test) # Make predictions on the test data

mse = mean_squared_error(y_test, y_pred) # Calculate the Mean Squared Error (MSE)
print(f"Mean Squared Error (MSE): {mse}") # Print the MSE

# Optional: Visualize the result
plt.scatter(X_test, y_test, color='blue', label='True values')
plt.plot(X_test, y_pred, color='red', label='Predicted values')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression - True vs Predicted Values")
plt.legend()
plt.show()

# Output => Mean Squared Error (MSE): 104.20222653187027