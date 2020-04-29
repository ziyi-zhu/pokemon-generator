import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers

def make_generator_model():
	model = tf.keras.Sequential()
	model.add(layers.Dense(7*7*512, use_bias=False, input_shape=(100,)))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Reshape((7, 7, 512)))
	assert model.output_shape == (None, 7, 7, 512) # Note: None is the batch size

	model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
	assert model.output_shape == (None, 7, 7, 256)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
	assert model.output_shape == (None, 14, 14, 128)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
	assert model.output_shape == (None, 28, 28, 3)

	return model

generator = make_generator_model()

generator.summary()

def make_discriminator_model():
	model = tf.keras.Sequential()
	model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same',
	                                 input_shape=[28, 28, 3]))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))

	model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))

	model.add(layers.Flatten())
	model.add(layers.Dense(1))

	return model

discriminator = make_discriminator_model()

discriminator.summary()

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	total_loss = real_loss + fake_loss
	return total_loss

def generator_loss(fake_output):
	return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'

checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator,
)

latest = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint.restore(latest)

noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

def generate_and_show_images(model, test_input):
	# Notice `training` is set to False.
	# This is so all layers run in inference mode (batchnorm).
	predictions = model(test_input, training=False)

	fig = plt.figure(figsize=(4,4))

	for i in range(predictions.shape[0]):
		plt.subplot(4, 4, i + 1)
		plt.imshow(predictions[i] * 0.5 + 0.5)
		plt.axis('off')

	plt.show()

generate_and_show_images(generator, seed)

# generator.save('saved_model/generator.h5') 
