import numpy as np
import matplotlib.pyplot as plt

# Binary Cross-Entropy Loss function
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Avoid log(0)

    # Compute Binary Cross-Entropy Loss
    loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# Generate a range of predicted probabilities from 0 to 1
predicted_probs = np.linspace(0, 1, 10)

# Calculate Binary Cross-Entropy Loss for y_true = 1 and y_true = 0
loss_true_1 = binary_cross_entropy(1, predicted_probs)
loss_true_0 = binary_cross_entropy(0, predicted_probs)

print(f"Binary Cross-Entropy Loss for true_1, {loss_true_1}")
print(f"Binary Cross-Entropy Loss for true_0, {loss_true_0}")

# Plot the graph
plt.figure(figsize=(8, 6))
plt.plot(predicted_probs, loss_true_1, label='y_true = 1', color='blue')
plt.plot(predicted_probs, loss_true_0, label='y_true = 0', color='red')

# Adding labels and title
plt.title('Binary Cross-Entropy Loss (Log Loss) vs Predicted Probability')
plt.xlabel('Predicted Probability')
plt.ylabel('Cross-Entropy Loss')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

# Output
# Binary Cross-Entropy Loss for true_1, [3.45387764e+01 2.19722458e+00 1.50407740e+00 1.09861229e+00
#  8.10930216e-01 5.87786665e-01 4.05465108e-01 2.51314428e-01
#  1.17783036e-01 9.99200722e-16]
# Binary Cross-Entropy Loss for true_0, [9.99200722e-16 1.17783036e-01 2.51314428e-01 4.05465108e-01
#  5.87786665e-01 8.10930216e-01 1.09861229e+00 1.50407740e+00
#  2.19722458e+00 3.45395760e+01]