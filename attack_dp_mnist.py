import argparse
from third_party.carlini.l2_attack import CarliniL2

import os
import numpy as np
import tensorflow as tf

IMAGE_SIZE = 28

def load_trained_mnist(ckpt, inputs, suffix=''):
    suffix = str(suffix)
    if suffix:
        suffix = '_' + suffix
    importer = tf.train.import_meta_graph(ckpt + '.meta', import_scope='mnist' + suffix,
                                          input_map={'dp_mnist_input': inputs})
    output = tf.get_collection('dp_mnist_output')[0]

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

def main():
    with tf.Session() as sess:
        model = Model("trained/dp_mnist")
        all_ones = np.ones((1, IMAGE_SIZE, IMAGE_SIZE)) * 0.1
        dummy_input = tf.constant(all_ones, dtype=tf.float32)

        my_data = np.genfromtxt('newimg.csv', delimiter=',')
        my_data.reshape(1, 28, 28, 1)

        plc = tf.placeholder_with_default(tf.zeros((1, 28, 28, 1), dtype=tf.float32), shape=(None, 28, 28, 1),
                                          name="side_in")

        mnist_output = model.predict(plc)

        def run_model(the_inputs):
            the_inputs = the_inputs.reshape((-1, 28, 28, 1))
            print(the_inputs)
            output = sess.run([mnist_output], feed_dict={plc: the_inputs})
            print('Output vector:', output[0])
            print('Classification:', np.argmax(output[0][0]))
            return output

        run_model(my_data)

        # output = model.predict(tf.reshape(tf.convert_to_tensor(my_data), (1, 28, 28, 1)))
        # print(output.eval())


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
