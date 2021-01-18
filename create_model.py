import tensorflow as tf
import warnings

warnings.simplefilter(action = 'ignore', category = FutureWarning)

# Set the path to training data
train_path = 'archive/data/train'

# Preprocess and load images
train_batches = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory = train_path, target_size = (244, 244), classes = ['alien', 'predator'], batch_size = 10)

# Create a callback to stop training if accuracy reaches 80% in order to prevent the model from overfitting
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if(logs.get('accuracy') > 0.8):
            print("\nReached 80% accuracy so stopping training!!")
            self.model.stop_training = True

# Initialize the callback
callbacks = myCallback()

# Create the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size = (3, 3), activation = 'relu', padding = 'same', input_shape = (244, 244, 3)),
    tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = 2),
    tf.keras.layers.Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
    tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2, activation = 'softmax')
])

# Print summary of the model
print("Model structure:\n")
print(model.summary())

# Compile the model
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), loss = tf.keras.losses.CategoricalCrossentropy(), metrics = ['accuracy'])

# Fit the model with training dataset
model.fit(x = train_batches, epochs = 10, verbose = 2, callbacks = [callbacks])

# Save the model
model.save("tf_model.h")