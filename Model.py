from third_party.carlini.l2_attack import CarliniL2
from third_party.differential_privacy.dp_sgd.dp_mnist import Eval_one_no_softmax
from third_party.differential_privacy.dp_sgd.dp_optimizer import utils
import sys
from six.moves import xrange


class Model:
    def __init__(self, save_path):

        self.eval_data_path = "data/mnist_train.tfrecord"
        self.image_size = 28
        self.num_channels = 1
        self.num_labels = 10
        self.NUM_TESTING_IMAGES = 10000
        self.network_parameters = utils.NetworkParameters()
        self.network_parameters.input_size = self.image_size ** 2
        self.network_parameters.default_gradient_l2norm_bound = 4.0
        self.save_path = save_path
        self.num_conv_layers = 0
        self.projection_dimensions = 60
        self.freeze_bottom_layers = False
        self.num_hidden_layers = 1
        self.hidden_layer_num_units = 1000

        if self.num_conv_layers > 0:
            conv = utils.ConvParameters()
            conv.name = "conv1"
            conv.in_channels = 1
            conv.out_channels = 128
            conv.num_outputs = 128 * 14 * 14
            self.network_parameters.conv_parameters.append(conv)
            # defaults for the rest: 5x5,stride 1, relu, maxpool 2x2,stride 2.
            # insize 28x28, bias, stddev 0.1, non-trainable.
        if self.num_conv_layers > 1:
            conv = self.network_parameters.ConvParameters()
            conv.name = "conv2"
            conv.in_channels = 128
            conv.out_channels = 128
            conv.num_outputs = 128 * 7 * 7
            conv.in_size = 14
            # defaults for the rest: 5x5,stride 1, relu, maxpool 2x2,stride 2.
            # bias, stddev 0.1, non-trainable.
            self.network_parameters.conv_parameters.append(conv)

        if self.num_conv_layers > 2:
            raise ValueError("Currently --num_conv_layers must be 0,1 or 2."
                             "Manually create a network_parameters proto for more.")

        if self.projection_dimensions > 0:
            self.network_parameters.projection_type = "PCA"
            self.network_parameters.projection_dimensions = self.projection_dimensions
        for i in xrange(self.num_hidden_layers):
            hidden = utils.LayerParameters()
            hidden.name = "hidden%d" % i
            hidden.num_units = self.hidden_layer_num_units
            hidden.relu = True
            hidden.with_bias = False
            hidden.trainable = not self.freeze_bottom_layers
            self.network_parameters.layer_parameters.append(hidden)

        logits = utils.LayerParameters()
        logits.name = "logits"
        logits.num_units = 10
        logits.relu = False
        logits.with_bias = False
        self.network_parameters.layer_parameters.append(logits)

    def predict(self, image):

        prediction = Eval_one_no_softmax(image,
                                         self.network_parameters,
                                         1,
                                         randomize=False, load_path=self.save_path,
                                         save_mistakes=False)
        print('[%s]' % ',\n'.join(map(str, prediction)))
        return prediction

    def batch_predict(self):

        prediction = Eval_one_no_softmax(self.eval_data_path,
                                         self.network_parameters,
                                         num_testing_images=self.NUM_TESTING_IMAGES,
                                         randomize=False, load_path=self.save_path,
                                         save_mistakes=False)
        print('[%s]' % ',\n'.join(map(str, prediction)))
        return prediction
