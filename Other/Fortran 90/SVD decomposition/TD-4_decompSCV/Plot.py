import numpy as np
import matplotlib.pyplot as plt

d = np.loadtxt("Initdat.dat")
p = np.loadtxt("Decompdat.dat")
plt.scatter(d[:, 0], d[:, 1], label = "Données")
plt.plot(p[:, 0], p[:, 1], label = "Decomp SVD", color = "red")
plt.legend()
plt.savefig("SVD.jpg")
plt.show()
