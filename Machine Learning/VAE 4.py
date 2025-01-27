import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Sampling function for latent space
def z_sampling(args, latent_dim):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    gaussian = tf.random.normal(shape=(batch, latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * gaussian

# Custom VAE Loss Function
def vae_loss(x, x_hat, z_mean, z_log_var):
    J_content = tf.reduce_mean(binary_crossentropy(x, x_hat))
    D_kl = tf.square(z_mean) + tf.exp(z_log_var) - z_log_var - 1
    J_kl = 0.5 * tf.reduce_sum(D_kl, axis=-1)
    return J_content + J_kl

# Encoder using your CNN structure
def build_encoder(input_shape, latent_dim):
    input_img = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.001))(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    z_mean = Dense(latent_dim, activation='linear')(x)
    z_log_var = Dense(latent_dim, activation='linear')(x)
    z = Lambda(z_sampling, output_shape=(latent_dim,), arguments={'latent_dim': latent_dim})([z_mean, z_log_var])
    return Model(inputs=input_img, outputs=[z_mean, z_log_var, z])


# Decoder
def build_decoder(output_shape, latent_dim):
    decoder_input = Input(shape=(latent_dim,))
    x = Dense(4 * 4 * 256, activation='relu')(decoder_input)
    x = Reshape((4, 4, 256))(x)
    x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)
    x = BatchNormalization()(x)
    output_img = Conv2DTranspose(output_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)
    return Model(inputs=decoder_input, outputs=output_img)


def main():
    start_time = time.time()
    
    # Load CIFAR-10 dataset
    (X_train, _), (X_test, _) = cifar10.load_data()
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    input_shape = (32, 32, 3)
    
    # Convert to grayscale if needed
    def convert_to_grayscale(images):
        return np.dot(images[...,:3], [0.2989, 0.5870, 0.1140]).reshape(-1, 32, 32, 1)
    
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
    color_modes = ['color', 'grayscale']
    latent_dims = [16, 32, 64, 128]
    batch_sizes = [16, 32, 64, 128]
    epochs_list = [20, 50, 100,1000]
    
    total_iterations = len(learning_rates) * len(color_modes) * len(latent_dims) * len(batch_sizes) * len(epochs_list)
    with tqdm(total=total_iterations, desc="Training VAE") as pbar, open('vae_results_8.txt', 'a') as f:
        for color_mode in color_modes:
            if color_mode == 'grayscale':
                X_train_mode = convert_to_grayscale(X_train)
                X_test_mode = convert_to_grayscale(X_test)
                input_shape_mode = (32, 32, 1)
            else:
                X_train_mode = X_train
                X_test_mode = X_test
                input_shape_mode = input_shape
            
            for latent_dim in latent_dims:
                for lr in learning_rates:
                    for batch_size in batch_sizes:
                        for epochs in epochs_list:
                            
                                encoder = build_encoder(input_shape_mode, latent_dim)
                                decoder = build_decoder(input_shape_mode, latent_dim)
                                
                                input_img = Input(shape=input_shape_mode)
                                z_mean, z_log_var, z = encoder(input_img)
                                output_img = decoder(z)
                                
                                vae = Model(inputs=input_img, outputs=output_img)
                                
                                optimizer = Adam(learning_rate=lr)
                                
                                def vae_loss_wrapper(x, x_hat):
                                    z_mean, z_log_var, _ = encoder(x)
                                    return vae_loss(x, x_hat, z_mean, z_log_var)
                                
                                vae.compile(optimizer=optimizer, loss=vae_loss_wrapper)
                                
                                history = vae.fit(X_train_mode, X_train_mode,
                                                  epochs=epochs,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  validation_data=(X_test_mode, X_test_mode),
                                                  callbacks=[EarlyStopping(patience=10)])
                                
                                # Plotting the loss
                                plt.plot(history.history['loss'], label='Training Loss')
                                plt.plot(history.history['val_loss'], label='Validation Loss')
                                plt.xlabel('Epochs')
                                plt.ylabel('Loss')
                                plt.legend()
                                plt.title(f'Loss During Training (LR={lr}, Latent Dim={latent_dim}, Mode={color_mode}, Batch Size={batch_size}, Epochs={epochs})')
                                plt.savefig(f'vae_training_loss_lr{lr}_latent{latent_dim}_mode{color_mode}_batch{batch_size}_epochs{epochs}_8.png')
                                plt.close()
                                
                                # Record the final results
                                final_loss = history.history['loss'][-1]
                                num_epochs = len(history.history['loss'])
                                iteration_time = time.time() - start_time

                                f.write(f'Epochs: {num_epochs}, Color Mode: {color_mode}, Latent Dim: {latent_dim}, Learning Rate: {lr}, Batch Size: {batch_size}, Epochs: {epochs}, Final Loss: {final_loss}, Time: {iteration_time:.2f} seconds\n')
                                f.flush()  # Force write to file
                                
                                # Save the model
                                vae.save(f'vae_cifar10_lr{lr}_latent{latent_dim}_mode{color_mode}_batch{batch_size}_epochs{epochs}.keras')
                                
                                # Generate and save reconstruction images
                                img_original = X_test_mode[0]
                                img_reconstructed = vae.predict(np.expand_dims(img_original, axis=0))[0]
                                
                                plt.figure(figsize=(10, 5))
                                plt.subplot(1, 2, 1)
                                plt.imshow(img_original.squeeze(), cmap='gray' if color_mode == 'grayscale' else None)
                                plt.title("Original Image")
                                plt.axis('off')
                                plt.subplot(1, 2, 2)
                                plt.imshow(img_reconstructed.squeeze(), cmap='gray' if color_mode == 'grayscale' else None)
                                plt.title("Reconstructed Image")
                                plt.axis('off')
                                plt.savefig(f'vae_reconstruction_lr{lr}_latent{latent_dim}_mode{color_mode}_batch{batch_size}_epochs{epochs}_8.png')
                                plt.close()
                                
                                pbar.update(1)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Training completed in: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()

