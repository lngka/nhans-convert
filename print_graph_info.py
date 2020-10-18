import os
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

trained_checkpoint_prefix = './trained_model/81448_0-1000000'
saved_model_dir = os.path.join('saved_model', '0')

graph = tf.Graph()
with tf.compat.v1.Session(graph=graph) as sess:
    tf.compat.v1.enable_resource_variables()

    # Restore from checkpoint
    loader = tf.compat.v1.train.import_meta_graph(
        trained_checkpoint_prefix + '.meta')
    loader.restore(sess, trained_checkpoint_prefix)

    for op in graph.get_operations():
        if op.type == "Placeholder":
            print(op)
        
        if op.name == "add_72":
            print('Outputs op: ')
            print(op)

    tf.compat.v1.train.write_graph(sess.graph_def, "./trained_model", "saved_graph.pb", as_text=False)
    tf.compat.v1.train.write_graph(sess.graph_def, "./trained_model", "saved_graph.pbtxt", as_text=True)
