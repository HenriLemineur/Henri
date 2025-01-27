import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Étape 1 : Charger le fichier .npz
file_path = '17_macor_500_500.npz'  # Remplace par le chemin vers ton fichier
data = np.load(file_path)

# Étape 2 : Extraire les données
RT_data = data['RT']  # Coefficients de réflexion et transmission
configs_data = data['configs']  # Configurations
args_data = data['args']  # Arguments passés

# Étape 3 : Extraire réflexions et transmissions
reflections = RT_data[..., 0]  # Coefficients de réflexion
transmissions = RT_data[..., 1]  # Coefficients de transmission

# Obtenir les valeurs des configurations pour les axes des graphiques
angles = np.unique(configs_data[:, 0])  # Angle de twist (en radians)
frequencies = np.unique(configs_data[:, 1])  # Fréquence

# Conversion des angles de radians à degrés
angles_degrees = np.degrees(angles)  # Conversion en degrés

# Étape 4 : Tracer les réflexions et les transmissions
plt.figure(figsize=(10, 12))

# Tracer les réflexions
plt.subplot(2, 1, 1)  # 2 lignes, 1 colonne, 1ère sous-figure
plt.imshow(reflections[:, :, 0, 0, 0, 0].T, aspect='auto', origin='lower', 
           extent=[angles_degrees.min(), angles_degrees.max(), frequencies.min(), frequencies.max()])
plt.colorbar(label='Réflexion')
plt.title('Réflexion en fonction de l\'angle de twist et de la fréquence')
plt.xlabel('Angle de Twist (degrés)')
plt.ylabel('Fréquence')

# Personnaliser les axes pour afficher les angles en degrés
plt.xticks(ticks=np.linspace(angles_degrees.min(), angles_degrees.max(), num=10))  # 10 ticks pour les angles
plt.yticks(ticks=np.linspace(frequencies.min(), frequencies.max(), num=10))  # 10 ticks pour les fréquences

# Formatage des ticks sur l'axe y pour afficher moins de décimales
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Affiche 2 décimales sur l'axe y

# Tracer les transmissions
plt.subplot(2, 1, 2)  # 2 lignes, 1 colonne, 2ème sous-figure
plt.imshow(transmissions[:, :, 0, 0, 0, 0].T, aspect='auto', origin='lower', 
           extent=[angles_degrees.min(), angles_degrees.max(), frequencies.min(), frequencies.max()])
plt.colorbar(label='Transmission')
plt.title('Transmission en fonction de l\'angle de twist et de la fréquence')
plt.xlabel('Angle de Twist (degrés)')
plt.ylabel('Fréquence')

# Personnaliser les axes pour afficher les angles en degrés
plt.xticks(ticks=np.linspace(angles_degrees.min(), angles_degrees.max(), num=10))  # 10 ticks pour les angles
plt.yticks(ticks=np.linspace(frequencies.min(), frequencies.max(), num=10))  # 10 ticks pour les fréquences

# Formatage des ticks sur l'axe y pour afficher moins de décimales
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Affiche 2 décimales sur l'axe y

plt.tight_layout()

# Sauvegarder la figure
plt.savefig('reflections_transmissions_degrees_3D_17_500_500.png', dpi=300)  # Sauvegarde la figure avec une résolution de 300 dpi
print("Figure sauvegardée sous 'reflections_transmissions_degrees_3D.png'.")

# Afficher la figure
plt.show()
