import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. Generate synthetic data (with a linear relationship + some noise)
np.random.seed(42)

# Generate X values (100 samples)
X = np.sort(5 * np.random.rand(100, 1), axis=0)

# Generate the target values (y = 2 * X + 1 with added noise)
y = 2 * X + 1 + np.random.randn(100, 1)  # Linear relationship with Gaussian noise

# 2. Introduce outliers to the data
y[::10] = y[::10] + 10  # Every 10th value has a large outlier

# 3. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the Huber Regressor model
model = HuberRegressor()
model.fit(X_train, y_train)

# 5. Make predictions using the trained model
y_pred = model.predict(X_test)

# 6. Calculate the Mean Absolute Error (MAE) to evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

# 7. Visualize the results
plt.scatter(X_test, y_test, color='blue', label='True values')
plt.plot(X_test, y_pred, color='red', label='Huber Regression Line (Predicted)')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Huber Loss Regression with Outliers")
plt.legend()
plt.show()

# Output => Mean Absolute Error (MAE): 3.483117828288826