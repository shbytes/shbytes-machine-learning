# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 1. Create a synthetic linear dataset
np.random.seed(42)

# Generate X values (random data)
X = 2 * np.random.rand(100, 1)  # 100 samples between 0 and 2

# Create the linear target values (y = 4 + 3x + noise)
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relationship with some noise

# 2. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make predictions on the test data
y_pred = model.predict(X_test)

# 5. Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

# 6. Visualize the results
plt.scatter(X_test, y_test, color='blue', label='True values')
plt.plot(X_test, y_pred, color='red', label='Linear regression line (predicted)')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression - True vs Predicted Values")
plt.legend()
plt.show()

# Output => Mean Absolute Error (MAE): 0.5913425779189777