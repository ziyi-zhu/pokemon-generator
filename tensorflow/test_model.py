import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers

# Recreate the exact same model, including its weights and the optimizer
generator = tf.keras.models.load_model('saved_model/generator.h5')

# Show the model architecture
generator.summary()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0] * 0.5 + 0.5)
plt.show()