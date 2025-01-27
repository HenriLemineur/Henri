import os
import time
from tqdm import tqdm
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def main():
    start_time = time.time()
    
    batch_combined = unpickle('combined_data_batch')
    batch_test = unpickle('test_batch')

    X_train = batch_combined[b'data']
    Y_train = batch_combined[b'labels']

    X_test = batch_test[b'data']
    Y_test = batch_test[b'labels']

    meta = unpickle('batches.meta')
    label_names = meta[b'label_names']

    results = []

    learning_rates = [ 0.0001]  # Liste des learning rates à tester
    #0.5, 0.1, 0.005, 0.001, 0.0005, a remettre dans lr
    for lr in tqdm(learning_rates, desc="Testing different learning rates"):
        for mode in ['bw', 'color']:
            print(f"\nMode: {mode}, Learning Rate: {lr}")
            if mode == 'bw':
                # Convertir les images en niveaux de gris
                X_train_mode = np.mean(X_train.reshape(-1, 3, 32, 32), axis=1).reshape(-1, 1024)
                X_test_mode = np.mean(X_test.reshape(-1, 3, 32, 32), axis=1).reshape(-1, 1024)
                input_shape = (1024,)
            else:
                # Utiliser les images en couleur
                X_train_mode = X_train.reshape(50000, 3072)
                X_test_mode = X_test.reshape(10000, 3072)
                input_shape = (3072,)
            
            print("\nShape of loaded data : ", X_train_mode.shape)
            
            # Normalisation
            X_train_mode = X_train_mode / 255.
            X_test_mode = X_test_mode / 255.

            # Créer le modèle de l'autoencodeur
            input_img = Input(shape=input_shape)  
            
            # Encodeur
            encoded = Dense(512, activation='relu')(input_img)
            encoded = Dense(256, activation='relu')(encoded)
            encoded = Dense(128, activation='relu')(encoded)
            
            # Décodeur
            decoded = Dense(256, activation='relu')(encoded)
            decoded = Dense(512, activation='relu')(decoded)
            decoded = Dense(input_shape[0], activation='sigmoid')(decoded)

            autoencoder = Model(inputs=input_img, outputs=decoded)
            autoencoder.summary()

            # Compilation et ajustement de l'autoencodeur
            optimizer = Adam(learning_rate=lr)
            autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
            callbacks = [EarlyStopping(monitor='val_loss', patience=50)]
            history = autoencoder.fit(X_train_mode, X_train_mode, epochs=5000, batch_size=100, callbacks=callbacks, validation_split=0.2)

            # Sauvegarde du modèle
           
            autoencoder.save(f'my_model_{mode}_lr_{lr}_4.keras')

            # Sauvegarde de la structure du modèle et des valeurs de perte
            results.append({
                'mode': mode,
                'learning_rate': lr,
                'encoder_summary': encoded,
                'decoder_summary': decoded,
                'history': history.history
            })

            # Affichage de la courbe de la perte
            plt.figure()
            plt.plot(history.history['loss'], label='Perte sur le set d\'entraînement')
            plt.plot(history.history['val_loss'], label='Perte sur le set de validation')
            plt.xlabel('Épochs')
            plt.ylabel('Perte')
            plt.legend()
            plt.title(f'Courbe de la perte pendant l\'entraînement ({mode}, LR={lr})')
            plt.savefig(f"training_loss_curve_ae_{mode}_lr_{lr}_4.png")
            plt.close()
            
            # Reconstruction d'une image à partir de l'autoencodeur
            index = np.random.randint(len(X_test_mode))
            if mode == 'bw':
                image_originale = X_test_mode[index].reshape(32, 32)
                image_reconstruite = autoencoder.predict(np.array([X_test_mode[index]]))
                image_reconstruite = image_reconstruite.reshape(32, 32)
            else:
                image_originale = X_test_mode[index].reshape(3, 32, 32).transpose(1, 2, 0)
                image_reconstruite = autoencoder.predict(np.array([X_test_mode[index]]))
                image_reconstruite = image_reconstruite.reshape(3, 32, 32).transpose(1, 2, 0)

            # Affichage de l'image originale et de sa reconstruction
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image_originale, cmap='gray' if mode == 'bw' else None)
            plt.title("Image originale")
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(image_reconstruite, cmap='gray' if mode == 'bw' else None)
            plt.title("Image reconstruite")
            plt.axis('off')
            plt.savefig(f"reconstruction_ae_{mode}_lr_{lr}_4.png")

    end_time = time.time()
    total_time = end_time - start_time

    with open("training_results_4.txt", "w") as f:
        f.write(f"Total training time: {total_time:.2f} seconds\n\n")
        for result in results:
            f.write(f"Mode: {result['mode']}, Learning Rate: {result['learning_rate']}\n")
            f.write("Encoder Summary:\n")
            f.write(f"{result['encoder_summary']}\n\n")
            f.write("Decoder Summary:\n")
            f.write(f"{result['decoder_summary']}\n\n")
            f.write("\nTraining History:\n")
            for key, value in result['history'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

if __name__ == '__main__':
    main()
