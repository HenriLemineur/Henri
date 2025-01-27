import matplotlib.pyplot as plt
import csv

# Fonction pour lire le fichier CSV et extraire les données
def read_csv(file_path):
    entries = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Ignorer les 9 premières lignes
        for _ in range(9):
            next(reader)
        for row in reader:
            entries.append(int(row[0]))

    return entries

# Chemin vers votre fichier CSV
file_path = 'SpectreGE_h1_Edep.csv'

# Lire les données du fichier CSV
entries = read_csv(file_path)

# Définir les numéros de canal (1 à N)
channels = range(1, len(entries) + 1)

# Créer une figure et des sous-graphiques
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Graphique en échelle linéaire
ax1.plot(channels, entries, drawstyle='steps-mid', color='blue', linewidth=1.5)
ax1.set_xlabel('Canal')
ax1.set_ylabel('Nombre de coups')
ax1.set_title('Spectre de dépôt d\'énergie (Échelle linéaire)')
ax1.grid(True)

# Graphique en échelle logarithmique
ax2.plot(channels, entries, drawstyle='steps-mid', color='blue', linewidth=1.5)
ax2.set_yscale('log')
ax2.set_xlabel('Canal')
ax2.set_ylabel('Nombre de coups (échelle logarithmique)')
ax2.set_title('Spectre de dépôt d\'énergie (Échelle logarithmique)')
ax2.grid(True)

# Ajuster la mise en page et afficher les graphiques
plt.tight_layout()
plt.show()
