import matplotlib.pyplot as plt
import numpy as np

steps = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900])
losses = np.array([2.859000, 0.147600, 0.105300, 0.103900, 0.102600, 0.099100, 0.093400, 0.099500,
                   0.095300, 0.091800, 0.096400, 0.097300, 0.095800, 0.088900, 0.095900, 0.096000,
                   0.094600, 0.100300])

# Doğrusal loss grafiği
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(steps, losses, marker='o')
plt.title('Training Loss over Steps')
plt.xlabel('Step')
plt.ylabel('Training Loss')
plt.grid(True)

# Gradyan (loss türevi) hesaplama
gradient = np.gradient(losses, steps)

plt.subplot(1, 2, 2)
plt.plot(steps, gradient, marker='x', color='r')
plt.title('Gradient of Training Loss over Steps')
plt.xlabel('Step')
plt.ylabel('Loss Gradient')
plt.grid(True)

plt.tight_layout()
plt.show()
