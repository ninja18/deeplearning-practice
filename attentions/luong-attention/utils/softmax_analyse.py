import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    x = x - np.max(x)  # numerical stability
    e = np.exp(x)
    return e / np.sum(e)

# Base logits (dot products before scaling)
base_logits = np.array([1.0, 0.8, 0.3, -0.2])

scales = np.linspace(0.1, 50, 400)
max_probs = []
entropies = []

for s in scales:
    p = softmax(base_logits * s)
    max_probs.append(np.max(p))
    entropies.append(-np.sum(p * np.log(p + 1e-12)))

plt.figure()
plt.plot(scales, max_probs, label="Max softmax probability")
plt.plot(scales, entropies, label="Entropy")
plt.xlabel("Logit scale")
plt.ylabel("Value")
plt.legend()
plt.title("Softmax collapse with increasing logit scale")
plt.show()
