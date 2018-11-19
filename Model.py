import os

import tensorflow as tf

class Model:
    def __init__(self, metagraph_path):
        self.ckpt = os.path.join(metagraph_path, 'ckpt')
        self.image_size = 28
        self.num_channels = 1
        self.num_labels = 10

    def predict(self, input_tensor):
        # sess = tf.get_default_session()
        # importer = tf.train.import_meta_graph(self.ckpt + '.meta', import_scope='mnist',
        #                                       input_map={'dp_mnist_input': images})
        # importer.restore(sess, self.ckpt)
        # return tf.get_collection('logits')[0]

        with tf.Session() as sess:
            importer = tf.train.import_meta_graph(self.ckpt + '.meta', import_scope='mnist',
	                                          input_map={'dp_mnist_input': input_tensor})
            mnist_output = tf.get_collection('dp_mnist_output')[0]
            sess.run(tf.global_variables_initializer())

            importer.restore(sess, self.ckpt)
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            _, output = sess.run([init_op, mnist_output])
            return output
