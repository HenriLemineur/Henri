import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import imageio
import os
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

# Charger le fichier .npz
data = np.load('17_macor_500_500.npz')

# Extraire les données nécessaires
RT = data['RT']  # Matrice des intensités
configs = data['configs']  # Configurations utilisées pour la simulation

# Extraire les dimensions automatiquement
num_angles = RT.shape[0]  # Nombre d'angles
num_frequencies = RT.shape[1]  # Nombre de fréquences

# Préparer les fréquences et angles pour les tracés
frequencies = np.linspace(0.3, 0.7, num_frequencies)  # Modifiez en fonction des valeurs réelles si nécessaire
angles = np.linspace(0, 90, num_angles)  # Angles en degrés, vous pouvez ajuster cette plage

# Créer une fenêtre pour recueillir les données de l'utilisateur
def get_angle_range_duration_and_grid_size():
    angle_start = None
    angle_end = None
    duration = None
    grid_size = None

    def on_submit():
        nonlocal angle_start, angle_end, duration, grid_size
        try:
            angle_start = float(entry_start.get())
            angle_end = float(entry_end.get())
            duration = float(entry_duration.get())
            grid_size = entry_grid.get()
            if angle_start < -90 or angle_start > 90 or angle_end < -90 or angle_end > 90:
                raise ValueError("Les angles doivent être entre -90 et 90.")
            if duration <= 0:
                raise ValueError("La durée doit être positive.")
            if not grid_size or "x" not in grid_size:
                raise ValueError("La taille du réseau doit être au format 'NxN' (ex. 3x3, 5x5).")
            root.destroy()  # Fermer la fenêtre
        except ValueError as e:
            messagebox.showerror("Erreur", str(e))

    # Création de la fenêtre
    root = tk.Tk()
    root.title("Entrée des Angles, Durée et Taille du Réseau")
    
    tk.Label(root, text="Angle de départ (-90 à 90):").pack()
    entry_start = tk.Entry(root)
    entry_start.pack()
    
    tk.Label(root, text="Angle d'arrivée (-90 à 90):").pack()
    entry_end = tk.Entry(root)
    entry_end.pack()
    
    tk.Label(root, text="Durée entre les images (en secondes):").pack()
    entry_duration = tk.Entry(root)
    entry_duration.pack()

    tk.Label(root, text="Taille du réseau (ex. 3x3, 5x5):").pack()
    entry_grid = tk.Entry(root)
    entry_grid.pack()

    # Bouton de soumission
    submit_button = tk.Button(root, text="Confirmer", command=on_submit)
    submit_button.pack()

    root.mainloop()  # Lancer la boucle principale de Tkinter

    return angle_start, angle_end, duration, grid_size

# Créer le dossier pour stocker les images
def create_image_folder(angle_start, angle_end, grid_size):
    folder_name = f"images_{angle_start:.2f}_{angle_end:.2f}_{grid_size}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

# Créer les images et la vidéo
def plot_graphs_and_create_video(angle_start, angle_end, duration, grid_size):
    # Trouver tous les angles valides entre angle_start et angle_end
    valid_angles = angles[(angles >= angle_start) & (angles <= angle_end)]
    image_files = []

    # Utilisation de tqdm pour afficher la progression
    with tqdm(total=len(valid_angles), desc="Création des images", unit="image") as pbar:
        for angle_idx, angle_rad in enumerate(valid_angles):
            # Récupérer les valeurs de R et T pour l'angle actuel
            R_values = RT[angle_idx, :, 0, 0, 0, 0, 0]  # Réflexion (R)
            T_values = RT[angle_idx, :, 0, 0, 0, 0, 1]  # Transmission (T)

            # Tracer la figure avec les courbes superposées
            plt.figure(figsize=(10, 6))
            plt.plot(frequencies, R_values, label='Réflexion (R)', color='blue')
            plt.plot(frequencies, T_values, label='Transmission (T)', color='red')

            # Calculer la somme et ajuster les valeurs pour une meilleure visualisation
            sum_values = R_values + T_values
            plt.plot(frequencies, sum_values, label='R + T', color='green', linestyle='dashed')

            plt.title(f'Réflexion et Transmission pour {angle_rad:.2f}°')
            plt.xlabel('Fréquence (a.u.)')
            plt.ylabel('Intensité')
            plt.ylim(0, 1.1)  # Ajustement de la limite de l'axe des y
            
            # Positionner la légende de manière fixe
            plt.legend(loc='upper right')
            plt.grid()

            # Sauvegarder chaque figure dans le dossier
            folder_path = f'images_{angle_start:.2f}_{angle_end:.2f}_{grid_size}'
            image_file_path = os.path.join(folder_path, f'reflection_transmission_{angle_rad:.2f}.png')
            plt.savefig(image_file_path)
            plt.close()  # Fermer la figure pour éviter de la montrer
            image_files.append(image_file_path)

            # Incrémenter la barre de progression
            pbar.update(1)

    # Créer la vidéo
    video_name = f'reflection_transmission_{angle_start:.2f}_{angle_end:.2f}_{duration:.2f}s_{grid_size}.mp4'
    clip = ImageSequenceClip(image_files, fps=1/duration)  # FPS basés sur la durée spécifiée
    clip.write_videofile(video_name, codec='libx264', fps=24)  # Codec H.264

    print(f'Vidéo créée : {video_name}')
    print(f'Images sauvegardées dans le dossier : {folder_path}')

# Demander les angles de départ et d'arrivée ainsi que la durée
angle_start, angle_end, duration, grid_size = get_angle_range_duration_and_grid_size()
if angle_start is not None and angle_end is not None and duration is not None and grid_size is not None:
    folder_name = create_image_folder(angle_start, angle_end, grid_size)
    plot_graphs_and_create_video(angle_start, angle_end, duration, grid_size)
