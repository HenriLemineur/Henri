
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dropout, BatchNormalization, Dense, Input
from keras.models import Model
from keras.utils import to_categorical
from keras import optimizers, regularizers
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
import tensorflow as tf
import pickle

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

def train_model_with_lr(learning_rate):
    combined_data_file = 'combined_data_batch'  # Fichier combiné
    label_names_file = 'batches.meta'  # Fichier contenant les noms des classes

    # Charger les données combinées
    print(f"\nImportation des données combinées pour learning rate {learning_rate} ...\n")
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

    X_train = X_train.reshape(40000, 3072)
    X_val = X_val.reshape(10000, 3072)
    X_test = X_test.reshape(10000, 3072)
    
    Y_train_onehot = to_categorical(Y_train, num_classes=10)
    Y_val_onehot = to_categorical(Y_val, num_classes=10)
    Y_test_onehot = to_categorical(Y_test, num_classes=10)

    N, D = X_train.shape
    K = 10

    input_layer = Input(shape=(D,))
    x = Dense(200, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)

    x = Dense(200, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)

    x = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)

    x = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)

    output_layer = Dense(K, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.summary()

    # Définir l'optimiseur Adam avec le learning rate spécifique
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Mise en place de l'arrêt précoce
    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    history = model.fit(X_train, Y_train_onehot, validation_data=(X_val, Y_val_onehot), epochs=3000, batch_size=64, verbose=1, callbacks=[early_stopping])

    # Nombre d'époques réelles exécutées
    num_epochs = len(history.epoch)

    # Évaluation du modèle sur les données de test
    test_acc = np.mean(np.equal(np.argmax(Y_test_onehot, 1), np.argmax(model.predict(X_test), 1)))

    print("\nAccuracy on the test set: %.5f" % test_acc)

    # Graphique des pertes
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss on the Training Set & Validation Set")
    plt.savefig(f"DNN5_Loss_lr{learning_rate}_epochs{num_epochs}.png")
    plt.show(block=False)  # Afficher l'image sans bloquer
    plt.pause(2)  # Afficher pendant 2 secondes
    plt.close()  # Fermer la figure

    # Graphique des accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy on the Training Set & Validation Set")
    plt.savefig(f"DNN5_Accuracy_lr{learning_rate}_epochs{num_epochs}.png")
    plt.show(block=False)  # Afficher l'image sans bloquer
    plt.pause(2)  # Afficher pendant 2 secondes
    plt.close()  # Fermer la figure

    # Affichage des images de test avec leurs prédictions
    num_images = 10
    indices = np.random.choice(len(X_test), num_images, replace=False)
    images = X_test[indices].reshape(num_images, 32, 32, 3).astype(np.float32) + 0.5  # Inverse de la normalisation
    predictions = model.predict(X_test[indices])
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = Y_test[indices]

    plt.figure(figsize=(15, 6))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        plt.title(f"Pred: {class_names[predicted_labels[i]]}\nTrue: {class_names[true_labels[i]]}")
        plt.axis('off')
    plt.tight_layout()

    # Sauvegarde de la figure des prédictions sur les images de test
    predictions_filename = f"DNN5_Predictions_lr{learning_rate}_epochs{num_epochs}.png"
    plt.savefig(predictions_filename)
    plt.show(block=False)  # Afficher l'image sans bloquer
    plt.pause(2)  # Afficher pendant 2 secondes
    plt.close()  # Fermer la figure

    print(f"Figure des prédictions sauvegardée sous: {predictions_filename}")

    # Création d'un fichier journal avec les paramètres et les performances du modèle
    summary_filename = f"DNN5_model_summary_lr{learning_rate}_epochs{num_epochs}.txt"
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Batch size: 64\n")
        f.write(f"Optimizer: Adam\n")
        f.write("\n")
        f.write("Accuracy on training set: {:.5f}\n".format(history.history['accuracy'][-1]))
        f.write("Accuracy on validation set: {:.5f}\n".format(history.history['val_accuracy'][-1]))
        f.write("Accuracy on test set: {:.5f}\n".format(test_acc))
        f.write("\n")
        f.write("Model architecture:\n")
        for layer in model.layers:
            if isinstance(layer, Dense):
                f.write(f"Layer: {layer.name}\n")
                f.write(f"\tUnits: {layer.units}\n")
        model.summary(print_fn=lambda x: f.write(x + '\n')) 

if __name__ == "__main__":
    learning_rates = [5.E-6,1.E-6, 5.E-7, 1.E-7]  # Liste des learning rates à tester
    for lr in learning_rates:
        train_model_with_lr(lr)





