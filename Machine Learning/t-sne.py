import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import time

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def rgb2gray(rgb):
    r = rgb[:, :1024]
    g = rgb[:, 1024:2048]
    b = rgb[:, 2048:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def main():
    # Fixer la graine pour la reproductibilité
    np.random.seed(0)

    # Charger les fichiers de batch combinés et de test
    batch_combined = unpickle('combined_data_batch')
    batch_test = unpickle('test_batch')

    # Extraire les données et les labels
    X_train = np.array(batch_combined[b'data'])
    Y_train = np.array(batch_combined[b'labels'])

    X_test = np.array(batch_test[b'data'])
    Y_test = np.array(batch_test[b'labels'])

    # Charger les noms des labels à partir du fichier meta
    meta = unpickle('batches.meta')
    label_names = meta[b'label_names']

    print("Forme des données chargées :", X_train.shape)
    
    # Convertir en niveaux de gris
    X_train_gray = rgb2gray(X_train)
    X_test_gray = rgb2gray(X_test)

    # Réduire la taille de l'échantillon
    sample_size = 10000
    X_train_gray = X_train_gray[:sample_size]
    Y_train = Y_train[:sample_size]
    X_train_color = X_train[:sample_size]  # Conserver également les données en couleur

    print("Nombre de données réduit à %d." % X_train_gray.shape[0])

    # Mélanger les données
    indices = np.arange(X_train_gray.shape[0])
    np.random.shuffle(indices)
    X_train_gray = X_train_gray[indices]
    X_train_color = X_train_color[indices]  # Mélanger également les données en couleur
    Y_train = Y_train[indices]

    # Normaliser les données
    scaler_gray = StandardScaler()
    X_train_gray = scaler_gray.fit_transform(X_train_gray)

    scaler_color = StandardScaler()
    X_train_color = scaler_color.fit_transform(X_train_color)

    # Initialiser t-SNE avec des paramètres optimisés
    perplexities = [70, 90, 110, 130, 150]
    max_iters = [5000, 7000, 9000, 11000, 13000, 21000]
    learning_rates = [50, 100, 200, 300, 500, 1000]

    best_score = -1
    best_config = None

    start_time = time.time()  # Capturer l'heure de début

    for perplexity in tqdm(perplexities, desc="Perplexities"):
        for max_iter in tqdm(max_iters, desc="Max Iterations", leave=False):
            for learning_rate in tqdm(learning_rates, desc="Learning Rates", leave=False):
                for data, color_mode in tqdm([(X_train_gray, 'gray'), (X_train_color, 'color')], desc="Color Modes", leave=False):
                    tsne = TSNE(perplexity=perplexity, max_iter=max_iter, learning_rate=learning_rate, verbose=0, n_jobs=-1, method='barnes_hut')

                    # Adapter t-SNE
                    Z = tsne.fit_transform(data)

                    # Calculer le score de silhouette pour évaluer la séparation des groupes
                    score = silhouette_score(Z, Y_train)

                    if score > best_score:
                        best_score = score
                        best_config = (perplexity, max_iter, learning_rate, color_mode)

                    # Tracer les résultats avec une carte de couleurs distincte
                    plt.figure(figsize=(10, 10))  # Augmenter la taille de la figure pour une meilleure visibilité
                    scatter = plt.scatter(Z[:, 0], Z[:, 1], s=20, c=Y_train, alpha=0.7, cmap='tab10')
                    plt.xlabel("$y_1$")
                    plt.ylabel("$y_2$")
                    plt.title(f"t-SNE sur les données CIFAR-10 ({color_mode}, perplexity={perplexity}, max_iter={max_iter}, learning_rate={learning_rate})")
                    
                    # Ajouter la légende des couleurs
                    handles, labels = scatter.legend_elements()
                    labels = [label.split('{')[-1].split('}')[0] for label in labels]
                    labels = [label_names[int(label)].decode('utf-8') for label in labels]
                    plt.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))

                    # Sauvegarder la figure
                    plt.savefig(f"tSNE_on_CIFAR-10_{color_mode}_perplexity_{perplexity}_max_iter_{max_iter}_learning_rate_{learning_rate}_12.png", bbox_inches='tight')
                    plt.close()  # Fermer toutes les figures pour permettre au code de continuer

    end_time = time.time()  # Capturer l'heure de fin
    elapsed_time = end_time - start_time  # Calculer le temps écoulé

    print(f"Temps total d'exécution: {elapsed_time:.2f} secondes")
    print(f"Meilleure configuration: perplexity={best_config[0]}, max_iter={best_config[1]}, learning_rate={best_config[2]}, color_mode={best_config[3]} avec un score de silhouette de {best_score:.4f}")

    # Sauvegarder la meilleure combinaison dans un fichier texte
    with open("best_tsne_config_12.txt", "w") as f:
        f.write(f"Temps total d'exécution: {elapsed_time:.2f} secondes\n")
        f.write(f"Meilleure configuration: perplexity={best_config[0]}, max_iter={best_config[1]}, learning_rate={best_config[2]}, color_mode={best_config[3]}\n")
        f.write(f"Score de silhouette: {best_score:.4f}\n")

if __name__ == '__main__':
    main()
