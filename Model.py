import os

import tensorflow as tf


def load_trained_mnist(ckpt, inputs, suffix=''):
    suffix = str(suffix)
    if suffix:
        suffix = '_' + suffix
    import_scope = 'mnist' + suffix

    importer = tf.train.import_meta_graph(ckpt + '.meta', import_scope=import_scope,
                                          input_map={'dp_mnist_input': inputs})
    outputs = tf.get_collection('dp_mnist_output')
    output = next(filter(lambda x: x.name.startswith(import_scope + '/'), reversed(outputs)))

    sess = tf.get_default_session()
    importer.restore(sess, ckpt)

    return output


def compute_model_path(model_path: str) -> str:
    ending = '.meta'
    if model_path.endswith(ending):
        model_path = model_path[:-len(ending)]
    return model_path


class Model:
    def __init__(self, model_path):
        self.model_path = compute_model_path(model_path)
        self.ckpt = os.path.join(self.model_path, 'ckpt')
        self.image_size = 28
        self.num_channels = 1
        self.num_labels = 10

        self.num_copies = 0

    def predict(self, input_tensor):
        self.num_copies = self.num_copies + 1
        return load_trained_mnist(self.ckpt, input_tensor, self.num_copies)

    # def predict(self, input_tensor):
    #     # sess = tf.get_default_session()
    #     # importer = tf.train.import_meta_graph(self.ckpt + '.meta', import_scope='mnist',
    #     #                                       input_map={'dp_mnist_input': images})
    #     # importer.restore(sess, self.ckpt)
    #     # return tf.get_collection('logits')[0]

    #     with tf.Session() as sess:
    #         importer = tf.train.import_meta_graph(self.ckpt + '.meta', import_scope='mnist',
    #                                               input_map={'dp_mnist_input': input_tensor})
    #         mnist_output = tf.get_collection('dp_mnist_output')[0]
    #         sess.run(tf.global_variables_initializer())

    #         importer.restore(sess, self.ckpt)
    #         init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #         _, output = sess.run([init_op, mnist_output])

    #         return tf.convert_to_tensor(output)
