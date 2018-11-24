import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model

def compute_model_path(model_path: str) -> str:
    ending = '.meta'
    if model_path.endswith(ending):
        model_path = model_path[:-len(ending)]
    return model_path


class Model:
    def __init__(self, model_path):
        sess = tf.Session()
        self.model_path = compute_model_path(model_path)
        self.ckpt = os.path.join(self.model_path, 'ckpt')
        self.image_size = 28
        self.num_channels = 1
        self.num_labels = 10

        self.num_copies = 0

        if self.num_copies:
            self.num_copies = '_' + self.num_copies
        importer = tf.train.import_meta_graph(self.ckpt + '.meta')
        weightVars = tf.trainable_variables()

        init = tf.global_variables_initializer()
        sess.run(init)
        weights = [sess.run(weight) for weight in weightVars]
        weightsLayer0 = [weights[0]]
        weightsLayer1 = [weights[1]]

        model = Sequential()
        model.add(Dense(1000, use_bias=False, input_dim=28*28))
        model.add(Dense(10, use_bias=False, input_dim=1000))
        model.layers[0].set_weights(weightsLayer0)
        model.layers[1].set_weights(weightsLayer1)
        self.model = model

    def get_ckpt(self):
        return self.ckpt

    def predict(self, input_tensor):
        num_inputs = tf.shape(input_tensor)[0]
        input_tensor = tf.reshape(input_tensor, (num_inputs, self.image_size ** 2))
        output = self.model(input_tensor)
        return output

