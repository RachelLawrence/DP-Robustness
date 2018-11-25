import time

import numpy as np
import tensorflow as tf

from Model2 import Model
from third_party.carlini.l2_attack_orig import CarliniL2
# from third_party.carlini.setup_cifar import CIFAR, CIFARModel
from third_party.carlini.setup_mnist import MNIST, MNISTModel

with tf.Session() as sess:
    # model = Model("trained/dp_mnist")
    model = Model("dp_sgd_out/noPCA")
    
    my_data = np.genfromtxt('newimg.csv', delimiter=',')
    my_data.reshape(1, 28, 28, 1)

    plc = tf.placeholder_with_default(tf.zeros((1, 28, 28, 1), dtype=tf.float32), shape=(None, 28, 28, 1),
                                      name="side_in")

    mnist_output = model.predict(plc)

    def run_model(the_inputs):
        the_inputs = the_inputs.reshape((-1, 28, 28, 1))
        output = sess.run([mnist_output], feed_dict={plc: the_inputs})
        print('Output vector:', output[0])
        print('Classification:', np.argmax(output[0][0]))
        return output

    run_model(my_data)