import tensorflow as tf
import os

tf.compat.v1.enable_resource_variables()

saved_model_dir = os.path.join('saved_model', '0')
model = tf.keras.models.load_model(saved_model_dir)

model.summary()