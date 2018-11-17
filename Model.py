import os

import tensorflow as tf


class Model:
    def __init__(self, metagraph_path):
        self.ckpt = os.path.join(metagraph_path, 'ckpt')

    def predict(self, images):
        sess = tf.get_default_session()
        importer = tf.train.import_meta_graph(self.ckpt + '.meta', import_scope='mnist',
                                              input_map={'mnist_input': images})
        importer.restore(sess, self.ckpt)
        return tf.get_collection('logits')[0]
