import numpy as np
import matplotlib.pyplot as plt

# Hinge Loss function
def hinge_loss(y_true, y_pred):
    # Ensure y_true is either -1 or 1
    loss = np.maximum(0, 1 - y_true * y_pred)
    return loss

# Generate a range of predicted scores (raw model outputs) from -2 to 2
predicted_scores = np.linspace(-2, 2, 100)

# True labels for binary classification (-1 or 1)
y_true_class_1 = np.array([1] * 100)  # True label for class 1
y_true_class_2 = np.array([-1] * 100)  # True label for class -1

# Calculate Hinge Loss for both classes (Class 1 and Class -1)
loss_class_1 = hinge_loss(y_true_class_1, predicted_scores)
loss_class_2 = hinge_loss(y_true_class_2, predicted_scores)

print(f"Hinge Loss for loss_class_1, {loss_class_1}")
print(f"Hinge Loss for loss_class_2, {loss_class_2}")

# Plot the graph
plt.figure(figsize=(8, 4))

plt.plot(predicted_scores, loss_class_1, label='y_true = 1 (Class 1)', color='blue')
plt.plot(predicted_scores, loss_class_2, label='y_true = -1 (Class -1)', color='red')

# Adding labels and title
plt.title('Hinge Loss vs Predicted Score')
plt.xlabel('Predicted Score (Raw Model Output)')
plt.ylabel('Hinge Loss')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

# Output
# Hinge Loss for loss_class_1, [3.         2.95959596 2.91919192 2.87878788 2.83838384 2.7979798
#  2.75757576 2.71717172 2.67676768 2.63636364 2.5959596  2.55555556
#  2.51515152 2.47474747 2.43434343 2.39393939 2.35353535 2.31313131
#  2.27272727 2.23232323 2.19191919 2.15151515 2.11111111 2.07070707
#  2.03030303 1.98989899 1.94949495 1.90909091 1.86868687 1.82828283
#  1.78787879 1.74747475 1.70707071 1.66666667 1.62626263 1.58585859
#  1.54545455 1.50505051 1.46464646 1.42424242 1.38383838 1.34343434
#  1.3030303  1.26262626 1.22222222 1.18181818 1.14141414 1.1010101
#  1.06060606 1.02020202 0.97979798 0.93939394 0.8989899  0.85858586
#  0.81818182 0.77777778 0.73737374 0.6969697  0.65656566 0.61616162
#  0.57575758 0.53535354 0.49494949 0.45454545 0.41414141 0.37373737
#  0.33333333 0.29292929 0.25252525 0.21212121 0.17171717 0.13131313
#  0.09090909 0.05050505 0.01010101 0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.        ]