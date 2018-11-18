import argparse

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


def main():
    parser = argparse.ArgumentParser(description='Run the trained MNIST model')
    parser.add_argument('--model_path', help='The path to the trained checkpoint (sans .meta)')
    parser.add_argument('-n', '--num_tensors', dest='num_tensors', type=int, help='Number of tensors to run',
                        default=10)
    args = parser.parse_args()

    model_path = compute_model_path(args.model_path)

    with tf.Session() as sess:
        all_ones = np.ones((args.num_tensors, IMAGE_SIZE ** 2))
        dummy_input = tf.constant(all_ones, dtype=tf.float32)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        mnist_output = load_trained_mnist(model_path, dummy_input)
        _, output = sess.run([init_op, mnist_output])

        print(output)


if __name__ == '__main__':
    main()
