from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from tqdm import tqdm

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def rgb2gray(rgb):
    r, g, b = rgb[:, :1024], rgb[:, 1024:2048], rgb[:, 2048:]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def fit_pca(n_components, svd_solver, whiten, data, y_train):
    pca = PCA(n_components=n_components, svd_solver=svd_solver, whiten=whiten)
    Z = pca.fit_transform(data)
    if n_components == 2:
        # Assurez-vous que y_train est de la même taille que Z
        score = silhouette_score(Z, y_train) 
    else:
        score = None
    return score, Z

def main():
    np.random.seed(0)

    batch_combined = unpickle('combined_data_batch')
    batch_test = unpickle('test_batch')

    X_train = np.array(batch_combined[b'data'])
    Y_train = np.array(batch_combined[b'labels'])

    X_test = np.array(batch_test[b'data'])
    Y_test = np.array(batch_test[b'labels'])

    meta = unpickle('batches.meta')
    label_names = meta[b'label_names']

    X_train_gray = rgb2gray(X_train)
    X_test_gray = rgb2gray(X_test)

    sample_size = 10000
    X_train_gray = X_train_gray[:sample_size]
    Y_train = Y_train[:sample_size]

    indices = np.arange(X_train_gray.shape[0])
    np.random.shuffle(indices)
    X_train_gray = X_train_gray[indices]
    Y_train = Y_train[indices]

    scaler_gray = StandardScaler()
    X_train_gray = scaler_gray.fit_transform(X_train_gray)

    n_components_options = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    svd_solver_options = ['auto', 'full', 'arpack', 'randomized']
    whiten_options = [True, False]

    best_score = -1
    best_config = None

    start_time = time.time()

    for color_mode, data in [('gray', X_train_gray), ('color', X_train[:sample_size])]:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        for n_components in tqdm(n_components_options, desc=f"n_components ({color_mode})"):
            for svd_solver in tqdm(svd_solver_options, desc=f"svd_solver ({color_mode})", leave=False):
                for whiten in tqdm(whiten_options, desc=f"whiten ({color_mode})", leave=False):
                    # Adjust Y_train to match the size of data
                    current_y_train = Y_train

                    score, Z = fit_pca(n_components, svd_solver, whiten, data, current_y_train)

                    if score is not None and score > best_score:
                        best_score = score
                        best_config = (n_components, svd_solver, whiten, color_mode)

                    if n_components == 2:
                        plt.figure(figsize=(10, 10))
                        scatter = plt.scatter(Z[:, 0], Z[:, 1], s=20, c=current_y_train, alpha=0.7, cmap='tab10')
                        plt.xlabel("$y_1$")
                        plt.ylabel("$y_2$")
                        plt.title(f"PCA on CIFAR-10 ({color_mode}, n_components={n_components}, svd_solver={svd_solver}, whiten={whiten})")
                        
                        handles, labels = scatter.legend_elements()
                        labels = [label.split('{')[-1].split('}')[0] for label in labels]
                        labels = [label_names[int(label)].decode('utf-8') for label in labels]
                        plt.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))

                        plt.savefig(f"PCA_on_CIFAR-10_{color_mode}_n_components_{n_components}_svd_solver_{svd_solver}_whiten_{whiten}_2.png", bbox_inches='tight')
                        plt.close()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"Best configuration: n_components={best_config[0]}, svd_solver={best_config[1]}, whiten={best_config[2]}, color_mode={best_config[3]} with a silhouette score of {best_score:.4f}")

    with open("best_pca_config_2.txt", "w") as f:
        f.write(f"Total execution time: {elapsed_time:.2f} seconds\n")
        f.write(f"Best configuration: n_components={best_config[0]}, svd_solver={best_config[1]}, whiten={best_config[2]}, color_mode={best_config[3]}\n")
        f.write(f"Silhouette score: {best_score:.4f}\n")

if __name__ == '__main__':
    main()
