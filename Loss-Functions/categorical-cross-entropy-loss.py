import numpy as np
import matplotlib.pyplot as plt

# Categorical Cross-Entropy Loss function
def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Avoid log(0)
    loss = - np.sum(y_true * np.log(y_pred), axis=1)
    return loss

# Generate a range of predicted probabilities for 3 classes
# For simplicity, we assume y_true is a one-hot vector for class 1 (e.g., [0, 1, 0])
classes = 3
predicted_probs = np.linspace(0.01, 0.99, 10)  # Range for predicted probabilities

# Create y_true as one-hot encoded vectors for each class
y_true_class_1 = np.array([[0, 1, 0]] * 10)  # One-hot for class 1
y_true_class_2 = np.array([[0, 0, 1]] * 10)  # One-hot for class 2
y_true_class_3 = np.array([[1, 0, 0]] * 10)  # One-hot for class 3

# For each class, we create predicted probabilities where the sum across each row is 1
y_pred_class_1 = np.column_stack([1 - predicted_probs, predicted_probs, np.zeros(10)])  # Class 1 is correct
y_pred_class_2 = np.column_stack([predicted_probs, 2 - predicted_probs, np.zeros(10)])  # Class 2 is correct
y_pred_class_3 = np.column_stack([predicted_probs, np.zeros(10), 1 - predicted_probs])  # Class 3 is correct

# Calculate the Categorical Cross-Entropy Loss for each case
loss_class_1 = categorical_cross_entropy(y_true_class_1, y_pred_class_1)
loss_class_2 = categorical_cross_entropy(y_true_class_2, y_pred_class_2)
loss_class_3 = categorical_cross_entropy(y_true_class_3, y_pred_class_3)

print(f"Categorical Cross-Entropy Loss for loss_class_1, {loss_class_1}")
print(f"Categorical Cross-Entropy Loss for loss_class_2, {loss_class_2}")
print(f"Categorical Cross-Entropy Loss for loss_class_3, {loss_class_3}")

# Plot the graph
plt.figure(figsize=(8, 6))

plt.plot(predicted_probs, loss_class_1, label='y_true = [0, 1, 0] (Class 1)', color='blue')
plt.plot(predicted_probs, loss_class_2, label='y_true = [0, 0, 1] (Class 2)', color='red')
plt.plot(predicted_probs, loss_class_3, label='y_true = [1, 0, 0] (Class 3)', color='green')

# Adding labels and title
plt.title('Categorical Cross-Entropy Loss vs Predicted Probability')
plt.xlabel('Predicted Probability for Correct Class')
plt.ylabel('Cross-Entropy Loss')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

# Output
# Categorical Cross-Entropy Loss for loss_class_1, [4.60517019 2.12956593 1.47938478 1.08866196 0.80843334 0.58978867
#  0.41047765 0.25848292 0.12657154 0.01005034]
# Categorical Cross-Entropy Loss for loss_class_2, [34.53877639 34.53877639 34.53877639 34.53877639 34.53877639 34.53877639
#  34.53877639 34.53877639 34.53877639 34.53877639]
# Categorical Cross-Entropy Loss for loss_class_3, [4.60517019 2.12956593 1.47938478 1.08866196 0.80843334 0.58978867
#  0.41047765 0.25848292 0.12657154 0.01005034]