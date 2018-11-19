## train_models.py -- train the neural network models for attacking
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

import tensorflow as tf
from third_party.carlini.setup_mnist import MNIST
from third_party.carlini.setup_cifar import CIFAR

from third_party.differential_privacy.dp_sgd.dp_optimizer import DPGradientDescentOptimizer
from third_party.differential_privacy.dp_sgd.dp_optimizer import AmortizedGaussianSanitizer
from third_party.differential_privacy.privacy_accountant.tf import accountant

import os

def train(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1, init=None):

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                           logits=predicted/train_temp)
    # with tf.Session() as sess:

    """
    Standard neural network training procedure.
    """
    model = Sequential()

    print(data.train_data.shape)
    
    model.add(Conv2D(params[0], (3, 3),
                            input_shape=data.train_data.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(params[1], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(params[2], (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(params[3], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(params[4]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params[5]))
    model.add(Activation('relu'))
    model.add(Dense(10))
    
    if init != None:
        model.load_weights(init)

    eps_delta = [1.0, 1e-5]
    default_gradient_l2norm_bound = 4.0 # Took the default value from dp_mnist.py

    priv_accountant = accountant.AmortizedAccountant(data.train_data.shape[0])
    with_privacy = (eps_delta[0] > 0)
    target_eps = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    gaussian_sanitizer = AmortizedGaussianSanitizer(
            priv_accountant,
            [default_gradient_l2norm_bound / batch_size, True])

    # Define the DP optimizer, can vary epsilon and delta.

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = tf.train.GradientDescentOptimizer(0.01) # TODO: WHAT TO DO ABOUT MINIMIZE COST?
    dp_sgd = DPGradientDescentOptimizer(learning_rate=0.01, eps_delta = eps_delta, 
                                                    sanitizer=gaussian_sanitizer)
    
    # Can compile the model with the new optimizer. 
    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              nb_epoch=num_epochs,
              shuffle=True)
    

    if file_name != None:
        model.save(file_name)

    # spent_eps_deltas = priv_accountant.get_privacy_spent(
    #                     sess, target_eps=target_eps)
    # for spent_eps, spent_delta in spent_eps_deltas:
    #     print("spent privacy: eps %.4f delta %.5g\n" % (spent_eps, spent_delta))

    return model

def train_distillation(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1):
    """
    Train a network using defensive distillation.

    Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks
    Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, Ananthram Swami
    IEEE S&P, 2016.
    """
    if not os.path.exists(file_name+"_init"):
        # Train for one epoch to get a good starting point.
        train(data, file_name+"_init", params, 1, batch_size)
    
    # now train the teacher at the given temperature
    teacher = train(data, file_name+"_teacher", params, num_epochs, batch_size, train_temp,
                    init=file_name+"_init")

    # evaluate the labels at temperature t
    predicted = teacher.predict(data.train_data)
    with tf.Session() as sess:
        y = sess.run(tf.nn.softmax(predicted/train_temp))
        print(y)
        data.train_labels = y

    # train the student model at temperature t
    student = train(data, file_name, params, num_epochs, batch_size, train_temp,
                    init=file_name+"_init")

    # and finally we predict at temperature 1
    predicted = student.predict(data.train_data)

    print(predicted)
    
if not os.path.isdir('models_noDP_gdTest'):
    os.makedirs('models_noDP_gdTest')

# train(CIFAR(), "models_DP/cifar", [64, 64, 128, 128, 256, 256], num_epochs=50)
# TODO: CHANGE NUM_EPOCHS
train(MNIST(), "models_noDP_gdTest/mnist", [32, 32, 64, 64, 200, 200], num_epochs=3)

# train_distillation(MNIST(), "models_DP/mnist-distilled-100", [32, 32, 64, 64, 200, 200],
                   # num_epochs=5, train_temp=100)
# train_distillation(CIFAR(), "models_DP/cifar-distilled-100", [64, 64, 128, 128, 256, 256],
                   # num_epochs=50, train_temp=100)
