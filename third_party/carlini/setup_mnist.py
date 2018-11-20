## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import scipy as sc
import os
import pickle
import gzip
import urllib.request

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model

from third_party.differential_privacy.dp_sgd.dp_optimizer import DPGradientDescentOptimizer
from third_party.differential_privacy.dp_sgd.dp_optimizer import AmortizedGaussianSanitizer
from third_party.differential_privacy.dp_sgd.dp_optimizer import dp_pca
from third_party.differential_privacy.privacy_accountant.tf import accountant

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, 28, 28, 1)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)

def extract_data_PCA(data_dir, filename, num_images, batch_size=128):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists("%s/%s_pca.csv" % (data_dir, filename.split('.')[0].split('/')[1])):
        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(num_images*28*28)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = (data / 255) - 0.5
            data = data.reshape(num_images, 28*28)

            with tf.Session() as sess:
                eps_delta = [0.5, 0.005]                # Took the default value from dp_mnist.py
                default_gradient_l2norm_bound = 4.0     # Took the default value from dp_mnist.py
                priv_accountant = accountant.AmortizedAccountant(num_images)

                gaussian_sanitizer = AmortizedGaussianSanitizer(
                        priv_accountant,
                        [default_gradient_l2norm_bound / batch_size, True])

                init_ops = tf.global_variables_initializer()
                sess.run(init_ops)

                proj = dp_pca.ComputeDPPrincipalProjection(data, 576,  # Chose 64 as proj. dims. since Goodfellow use 60
                                sanitizer = gaussian_sanitizer, eps_delta=eps_delta, sigma=None).eval()
                proj = np.transpose(proj)
                data = data.reshape(num_images, 28*28, 1)
                data_projected = [np.matmul(proj, vector) for vector in data]

                # Fill in any nan values with 0
                data_projected = np.asarray([sc.stats.mstats.winsorize(image, 0.05) for image in data_projected])
                # Normalize each image to have values between -1.0 and 1.0 so we can use arctan
                data_projected = [image * 0.5/ np.max(np.abs(image)) for image in data_projected]
                data_projected = np.nan_to_num(np.asarray(data_projected))
                data_projected = data_projected.reshape(num_images, 576)
                np.savetxt("%s/%s_pca.csv" % (data_dir, filename.split('.')[0].split('/')[1]), data_projected, delimiter=",")
                data_projected = data_projected.reshape(num_images, 24, 24, 1)
                return data_projected

    else:
        data_projected = np.genfromtxt("%s/%s_pca.csv" % (data_dir, filename.split('.')[0].split('/')[1]), delimiter=",")
        data_projected = data_projected.reshape(num_images, 24, 24, 1)
        return data_projected

class MNIST:
    def __init__(self):
        if not os.path.exists("data"):
            os.mkdir("data")
            files = ["train-images-idx3-ubyte.gz",
                     "t10k-images-idx3-ubyte.gz",
                     "train-labels-idx1-ubyte.gz",
                     "t10k-labels-idx1-ubyte.gz"]
            for name in files:

                urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)

        train_data = extract_data("data/train-images-idx3-ubyte.gz", 60000)
        train_labels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
        self.test_data = extract_data("data/t10k-images-idx3-ubyte.gz", 10000)
        self.test_labels = extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)
        
        VALIDATION_SIZE = 5000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]

class MNIST_PCA:
    def __init__(self):
        if not os.path.exists("data"):
            os.mkdir("data")
            files = ["train-images-idx3-ubyte.gz",
                     "t10k-images-idx3-ubyte.gz",
                     "train-labels-idx1-ubyte.gz",
                     "t10k-labels-idx1-ubyte.gz"]
            for name in files:

                urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)

        train_data = extract_data_PCA("data_pca", "data/train-images-idx3-ubyte.gz", 60000)
        train_labels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
        self.test_data = extract_data_PCA("data_pca", "data/t10k-images-idx3-ubyte.gz", 10000)
        self.test_labels = extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)
        
        VALIDATION_SIZE = 5000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]


class MNISTModel:
    def __init__(self, restore, session=None, recompile=False, train_temp=1, 
                            eps_delta=[1.0, 1e-5], pca=False, num_images=60000, batch_size = 128):
        self.num_channels = 1
        self.num_labels = 10

        model = Sequential()

        if pca:
            self.image_size = 24
            model.add(Conv2D(32, (3, 3),
                            input_shape=(24, 24, 1)))
        else:
            self.image_size = 28
            model.add(Conv2D(32, (3, 3),
                            input_shape=(28, 28, 1)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dense(10))
        model.load_weights(restore)

        if recompile:
            def fn(correct, predicted):
                return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                           logits=predicted/train_temp) 

            default_gradient_l2norm_bound = 4.0     # Took the default value from dp_mnist.py
            priv_accountant = accountant.AmortizedAccountant(num_images)

            gaussian_sanitizer = AmortizedGaussianSanitizer(
                    priv_accountant,
                    [default_gradient_l2norm_bound / batch_size, True])

            dp_sgd = DPGradientDescentOptimizer(learning_rate=0.01, eps_delta=eps_delta, 
                                                    sanitizer=gaussian_sanitizer)
        
            model.compile(loss=fn,
                      optimizer=dp_sgd,
                      metrics=['accuracy'])

        self.model = model

    def predict(self, data):
        return self.model(data)
