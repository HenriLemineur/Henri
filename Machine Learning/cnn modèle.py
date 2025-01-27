import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Model
from keras.utils import to_categorical
from keras import optimizers, regularizers
from sklearn.utils import shuffle
import pickle
from keras.regularizers import l2
import sys
def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

def load_combined_data(file):
    data = unpickle(file)
    X = data[b'data']
    Y = data[b'labels']
    X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Reshape et transpose pour obtenir la forme correcte
    return X, np.array(Y)

def load_label_names(file):
    data = unpickle(file)
    return [label.decode('utf-8') for label in data[b'label_names']]

def main():
    combined_data_file = 'combined_data_batch'  # Fichier combiné
    label_names_file = 'batches.meta'  # Fichier contenant les noms des classes

    # Charger les données combinées
    print("\nImportation des données combinées ...\n")
    X_train, Y_train = load_combined_data(combined_data_file)
    
    # Charger les données de test
    test_batch = unpickle('test_batch')
    X_test = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    Y_test = np.array(test_batch[b'labels'])

    # Charger les noms des classes
    class_names = load_label_names(label_names_file)

    # Mélanger les données d'entraînement
    X_train, Y_train = shuffle(X_train, Y_train)
    X_val = X_train[-10000:]
    X_train = X_train[:-10000]
    Y_val = Y_train[-10000:]
    Y_train = Y_train[:-10000]

    print("Forme des données d'entraînement:", X_train.shape)  # (40000, 32, 32, 3)
    print("Forme des données de validation:", X_val.shape)  # (10000, 32, 32, 3)
    print("Forme des données de test:", X_test.shape)  # (10000, 32, 32, 3)

    # Normalisation des données d'entrée
    X_train = (X_train / 255.0) - 0.5
    X_val = (X_val / 255.0) - 0.5
    X_test = (X_test / 255.0) - 0.5

    # Encodage one-hot des étiquettes
    Y_train_onehot = to_categorical(Y_train, num_classes=10)
    Y_val_onehot = to_categorical(Y_val, num_classes=10)
    Y_test_onehot = to_categorical(Y_test, num_classes=10)

    i = Input(shape=(32, 32, 3))

    # Première couche de convolution
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.001))(i)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    # Deuxième couche de convolution
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    # Troisième couche de convolution
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    # Quatrième couche de convolution
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    # Aplatir les données
    x = Flatten()(x)

    # Couche entièrement connectée
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)

    # Couche de sortie pour la classification
    o = Dense(10, activation='softmax')(x)

    model = Model(inputs=i, outputs=o)
    model.summary()
   
    optimizer = optimizers.Adam(learning_rate=1.E-5)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Entraînement du modèle
    epochs = 500
    history = model.fit(X_train, Y_train_onehot, validation_data=(X_val, Y_val_onehot), epochs=epochs, batch_size=64, verbose=1)

    # Évaluation du modèle sur les données de test
    test_loss, test_acc = model.evaluate(X_test, Y_test_onehot)
    print("Précision sur les données de test:", test_acc)

    # Prédictions sur le jeu de test
    predictions = model.predict(X_test)

    # Tracer l'évolution de l'accuracy du jeu d'entraînement et de validation
    plt.plot(history.history['accuracy'], label='Accuracy (train)')
    plt.plot(history.history['val_accuracy'], label='Accuracy (validation)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Evolution')
    plt.legend()
    plt.savefig('accuracy_evolutiontest.png')
    plt.show()

    # Enregistrer les valeurs d'accuracy dans un fichier

    # Créer un fichier résumant les paramètres du modèle et les valeurs d'accuracy
    with open('model_summarytest.txt', 'w', encoding='utf-8') as f:

        f.write("Nombre d'époques: {}\n".format(epochs))
        f.write("Taille du batch: {}\n".format(64))  # Taille du batch utilisée pendant l'entraînement
        f.write("Optimiseur: Adam\n")  # Optimiseur utilisé
        f.write("Learning rate: {}\n".format(1.E-3))  # Taux d'apprentissage utilisé
        f.write("\n")
        f.write("Accuracy on training set: {}\n".format(history.history['accuracy'][-1]))
        f.write("Accuracy on validation set: {}\n".format(history.history['val_accuracy'][-1]))
        f.write("Accuracy on test set: {}\n".format(test_acc))
        f.write("\n")
        f.write("Architecture du modèle:\n")
        for layer in model.layers:
                if isinstance(layer, Conv2D):
                    f.write("Layer: {}\n".format(layer.name))
                    f.write("\tfilters: {}\n".format(layer.filters))
                    f.write("\tkernel_size: {}\n".format(layer.kernel_size))
                elif isinstance(layer, Dense):
                    f.write("Layer: {}\n".format(layer.name))
                    f.write("\tunits: {}\n".format(layer.units))

            # Rediriger la sortie vers le fichier ouvert
        sys.stdout = f
        # Afficher le résumé du modèle
        model.summary()
        # Restaurer la sortie standard
        sys.stdout = sys.__stdout__
        
    

if __name__ == "__main__":
    main()
