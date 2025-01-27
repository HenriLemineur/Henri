import matplotlib.pyplot as plt
import numpy as np

files = ["points1.dat", "points2.dat", "points3.dat"]
relerr_values = [1.0E-4, 1.0E-9, 1.0E-14]

plt.figure(figsize=(15, 5))

# Tracer les graphes dans les sous-graphiques
for i, file in enumerate(files):

    data = np.loadtxt(file)

    # Extraire x et f(x)
    x = data[:, 0]
    y = data[:, 1]

    plt.subplot(1, 3, i + 1)
    plt.plot(x, y, 'o', label='Évaluations de fun(x)')
    plt.xlabel('x')
    plt.ylabel('fun(x)')
    plt.title(f'Points d\'évaluation de fun(x) pour relerr = {relerr_values[i]}')
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()
