import argparse
from third_party.carlini.l2_attack import CarliniL2

import numpy as np
import tensorflow as tf

IMAGE_SIZE = 28


def load_trained_mnist(model_path, inputs):
    importer = tf.train.import_meta_graph(model_path + '.meta', import_scope='mnist',
                                          input_map={'dp_mnist_input': inputs})
    output = tf.get_collection('dp_mnist_output')[0]

    sess = tf.get_default_session()
    importer.restore(sess, model_path)

    return output


def compute_model_path(model_path: str) -> str:
    ending = '.meta'
    if model_path.endswith(ending):
        model_path = model_path[:-len(ending)]
    return model_path


class Model:
    def __init__(self, model_path, sess):
        self.model_path = compute_model_path(model_path)
        self.image_size = 28
        self.num_channels = 1
        self.num_labels = 10
        self.sess = sess

    def predict(self, input_tensor):
        with self.sess.as_default() as sess:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            mnist_output = load_trained_mnist(self.model_path, input_tensor)
            _, output = sess.run([init_op, mnist_output])
            return output


def main():
    with tf.Session() as sess:
        model = Model("./trained/dp_mnist/ckpt", sess)
        all_ones = np.ones((1, IMAGE_SIZE ** 2))
        dummy_input = tf.constant(all_ones, dtype=tf.float32)

        output = model.predict(dummy_input)
        print(output)


    # parser = argparse.ArgumentParser(description='Run the trained MNIST model')
    # parser.add_argument('--model_path', help='The path to the trained checkpoint (sans .meta)')
    # parser.add_argument('-n', '--num_tensors', dest='num_tensors', type=int, help='Number of tensors to run',
    #                     default=10)
    # args = parser.parse_args()
    #
    # model_path = compute_model_path(args.model_path)
    #
    # with tf.Session() as sess:
    #     all_ones = np.ones((args.num_tensors, IMAGE_SIZE ** 2))
    #     dummy_input = tf.constant(all_ones, dtype=tf.float32)
    #
    #     init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #     mnist_output = load_trained_mnist(model_path, dummy_input)
    #     _, output = sess.run([init_op, mnist_output])
    #
    #     print(output)


if __name__ == '__main__':
    main()
