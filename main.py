from third_party.carlini.l2_attack import CarliniL2
from third_party.differential_privacy.dp_sgd.dp_mnist import Eval_one_no_softmax
from third_party.differential_privacy.dp_sgd.dp_optimizer import utils
import sys
import numpy as np
import tensorflow as tf
from Model import Model

model = Model("../output/dp_sgd/dp_mnist")

shape = (1, model.image_size, model.image_size, model.num_channels)
img = tf.Variable(np.zeros(shape), dtype=tf.float32)
img = tf.cast(tf.image.decode_png(example["image/encoded"], channels=1),
                tf.float32)
img = tf.reshape(img, [model.image_size * model.image_size])
img /= 255
model.predict(img)