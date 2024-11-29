import numpy as np
import matplotlib.pyplot as plt


# Function to calculate Kullback-Leibler (KL) Divergence
def kl_divergence(p, q):
    # Clip q values to avoid division by zero and log(0)
    epsilon = 1e-10
    q = np.clip(q, epsilon, 1.0)

    # Calculate the KL Divergence: sum(P * log(P / Q))
    return np.sum(p * np.log(p / q))


# Example usage:
# True distribution P and predicted distribution Q
# Both distributions should sum to 1 (valid probability distributions)

p = np.array([0.4, 0.6])  # True distribution
q = np.array([0.5, 0.5])  # Predicted distribution

# Calculate KL Divergence
kl_div = kl_divergence(p, q)
print(f"KL Divergence between P and Q: {kl_div}")

# To visualize KL Divergence for different distributions, we can plot it
# Generate different probability distributions for Q
q_values = np.linspace(0.01, 0.99, 100)
kl_values = []

# Calculate KL Divergence for different values of q (second distribution)
for q_val in q_values:
    q = np.array([q_val, 1 - q_val])  # q is [q_val, 1 - q_val]
    kl_values.append(kl_divergence(p, q))

# Plot the results
plt.figure(figsize=(8, 4))
plt.plot(q_values, kl_values, label='KL Divergence between P and Q')
plt.title('KL Divergence vs q-value for Second Distribution Q')
plt.xlabel('q-value of second distribution Q')
plt.ylabel('KL Divergence')
plt.grid(True)
plt.legend()
plt.show()

# Output => KL Divergence between P and Q: 0.020135513550688863