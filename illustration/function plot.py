import matplotlib.pyplot as plt
import numpy as np
import math

x = np.linspace(-10, 10, 100)
z = 1 / (1 + np.exp(-x))

plt.figure()
plt.plot(x, z)
plt.xlabel("x")
plt.ylabel("Sigmoid(X)")

plt.axhline(0, color="black", alpha=0.3)
plt.axvline(0, color="black", alpha=0.3)


#plt.show()
plt.savefig(f'sigmoid(x).png')
