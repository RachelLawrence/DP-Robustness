import os

import numpy as np
import tensorflow as tf

from Model import Model

with tf.Session() as sess:
    model = Model(os.path.join('..', 'output', 'dp_sgd', 'dp_mnist'))

    shape = (1, 28 * 28)
    img = tf.constant(np.zeros(shape), dtype=tf.float32)

    prediction = model.predict(img)
    print(sess.run(prediction))

