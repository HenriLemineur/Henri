import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

data = pd.read_csv('231208_7.csv', delimiter=';')

QL = data.iloc[:, 3]
QS = data.iloc[:, 4]
PSD = (QL - QS) / QL

mask = (PSD >= 0.5) & (PSD <= 1)
filtered_QL = QL[mask]
filtered_PSD = PSD[mask]

xedges = np.linspace(min(filtered_QL), max(filtered_QL), 30)
yedges = np.linspace(min(filtered_PSD), max(filtered_PSD), 30)
H, xedges, yedges = np.histogram2d(filtered_QL, filtered_PSD, 
                                   bins=(xedges, yedges))

x_indices = np.clip(np.digitize(filtered_QL, xedges), 0, 
                    H.shape[0] - 1)
y_indices = np.clip(np.digitize(filtered_PSD, yedges), 0, 
                    H.shape[1] - 1)

colors = [(0, 'blue'), (0.25, 'cyan'), (0.35, '#00FF00'), 
          (0.65, 'yellow'), (1, 'red')]
cmap = LinearSegmentedColormap.from_list('CustomMap', colors)

plt.figure(figsize=(8, 6))
sc = plt.scatter(filtered_QL, filtered_PSD, c=H[x_indices, y_indices],
                  cmap=cmap, s=1)
plt.colorbar(sc, label='Densité de points')
plt.xlabel('Charge mesurée (u.a)')
plt.ylabel('PSD')
plt.title('Spectre PSD pour la détection de neutrons')
plt.grid(True)
plt.show()
