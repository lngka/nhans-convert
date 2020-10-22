import tensorflow as tf
import os

saved_model_dir = os.path.join('saved_model', '0')

# Convert the SavedModel model to tflite
converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(
    saved_model_dir, 
    input_arrays=['mixedph','noiseposcontextph', 'noisenegcontextph'], 
    output_arrays=['add_72'])  


#converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
with open('./n_hans.tflite', 'wb') as f:
    f.write(tflite_model)
