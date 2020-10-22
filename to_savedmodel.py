import os
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

trained_checkpoint_prefix = './trained_model/81448_0-1000000'
saved_model_dir = os.path.join('saved_model', '0') # was used to convert to tflite
#saved_model_dir = os.path.join('saved_model', '1') 


graph = tf.Graph()
with tf.compat.v1.Session(graph=graph) as sess:
    tf.compat.v1.enable_resource_variables()

    # Restore from checkpoint
    loader = tf.compat.v1.train.import_meta_graph(
        trained_checkpoint_prefix + '.meta')
    loader.restore(sess, trained_checkpoint_prefix)

    # Export checkpoint to SavedModel
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(saved_model_dir)

    noisenegcontextph = tf.compat.v1.placeholder("float32", name='noisenegcontextph', shape=(1, 200, 201))
    noiseposcontextph = tf.compat.v1.placeholder("float32", name='noiseposcontextph', shape=(1, 200, 201))
    mixedph = tf.compat.v1.placeholder("float32", name='mixedph', shape=(1, 200, 201))
    denoised = tf.compat.v1.placeholder("float32", name='add_72', shape=(1, 201))


    tinfo_noisenegcontextph = tf.compat.v1.saved_model.utils.build_tensor_info(noisenegcontextph)
    tinfo_noiseposcontextph = tf.compat.v1.saved_model.utils.build_tensor_info(noiseposcontextph)
    tinfo_mixedph = tf.compat.v1.saved_model.utils.build_tensor_info(mixedph)
    tinfo_denoised = tf.compat.v1.saved_model.utils.build_tensor_info(denoised)

    prediction_signature = (tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
                                inputs={
                                    'noisenegcontextph': tinfo_noisenegcontextph,
                                    'noiseposcontextph': tinfo_noiseposcontextph,
                                    'mixedph': tinfo_mixedph,
                                },
                                outputs={'add_72': tinfo_denoised},
                                method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(sess,
                                        tags = [tf.compat.v1.saved_model.SERVING],
                                        strip_default_attrs=True, 
                                        signature_def_map= {tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature })
    builder.save()
