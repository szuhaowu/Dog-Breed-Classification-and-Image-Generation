from tqdm import tqdm
import numpy as np

from adversarial_networks import create_generator, create_discriminator, create_gan
from data import generate_noise
from image import normalize_image
from plot import plot_images, plot_loss, plot_test


PLOT_FRECUENCY = 100


def training(x_train,x_test,epochs=1, batch_size=32):
    #Loading Data
    batches = x_train.shape[0] / batch_size

    # Creating GAN
    generator = create_generator()
    discriminator = create_discriminator()
    gan = create_gan(generator, discriminator)
    
    # Adversarial Labels
    y_valid = np.ones(batch_size)*0.9
    y_fake = np.zeros(batch_size)
    discriminator_loss, generator_loss = [], []

    for epoch in range(1, epochs+1):
        print('-'*15, 'Epoch', epoch, '-'*15)
        g_loss = 0; d_loss = 0

        for _ in tqdm(range(int(batches))):
            # Random Noise and Images Set
            noise = generate_noise(batch_size)
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # Generate Fake Images
            generated_images = generator.predict(noise)
            
            # Train Discriminator (Fake and Real)
            discriminator.trainable = True
            d_valid_loss = discriminator.train_on_batch(image_batch, y_valid)
            d_fake_loss = discriminator.train_on_batch(generated_images, y_fake)            

            d_loss += (d_fake_loss + d_valid_loss)/2
            
            # Train Generator
            noise = generate_noise(batch_size)
            discriminator.trainable = False
            g_loss += gan.train_on_batch(noise, y_valid)
            
        discriminator_loss.append(d_loss/batches)
        generator_loss.append(g_loss/batches)
            
        if epoch % PLOT_FRECUENCY == 0:
            plot_images(epoch, generator)
            plot_loss(epoch, generator_loss, discriminator_loss)
            plot_test(epoch, x_test, generator)
    
    generator.save('Save_model/generator.h5')
    discriminator.save('Save_model/discriminator.h5')
    gan.save('Save_model/gan.h5')

    #save_images(generator)


if __name__ == '__main__':
    #Load data
    x_train = np.load('../Save_Data/train_data_64.npy')
    x_train = x_train*2-1
    x_test = np.load('../Save_Data/yaya_64.npy')
    x_test = normalize_image(x_test)
    
    training(x_train,x_test,epochs=200)
    