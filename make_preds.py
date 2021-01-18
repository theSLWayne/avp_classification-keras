import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model("tf_model.h5")

# Set the path to test data
test_path = 'archive/data/test'

# Preprocess and load test images
test_batches = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory = test_path, target_size = (244, 244), classes = ['alien', 'predator'], batch_size = 10)

# Evaluate the model with test data
result = model.evaluate(test_batches)

# Print results
print("Test data loss: {}".format(result[0]))
print("Test data accuracy: {}".format(result[1]))