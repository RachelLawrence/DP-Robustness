import numpy as np
import tensorflow as tf

# Program constants

IMAGE_SIZE = 28
BATCH_SIZE = 600
LEARNING_RATE = 0.001


# Read the dataset

def mnist_parser(example):
    features = {"image/encoded": tf.FixedLenFeature(shape=(), dtype=tf.string),
                "image/class/label": tf.FixedLenFeature([1], tf.int64)}

    example = tf.parse_single_example(example, features=features)

    image = tf.cast(tf.image.decode_png(example["image/encoded"], channels=1),
                    tf.float32)
    image = tf.reshape(image, [IMAGE_SIZE * IMAGE_SIZE])
    image /= 255
    label = tf.cast(example["image/class/label"], dtype=tf.int32)
    label = tf.reshape(label, [])

    return image, label


def load_data():
    dataset = tf.data.TFRecordDataset(['data/mnist_train.tfrecord'])
    dataset = dataset.map(mnist_parser)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(2)
    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


# Create a very simple model with an input placeholder

def build_model(inputs):
    input = tf.placeholder_with_default(inputs, [None, IMAGE_SIZE ** 2], 'mnist_input')
    output = tf.layers.dense(input, 200, activation=tf.nn.relu)
    output = tf.layers.dense(output, 10, activation=tf.nn.relu)
    tf.add_to_collection('output', output)
    return output


def build_loss(logits, labels):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(labels, 10))
    loss = tf.reduce_mean(loss)
    return loss


# Train the model

def train():
    images, labels = load_data()
    logits = build_model(images)
    loss = build_loss(logits, labels)
    gd_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        try:
            step = 0
            while True:
                _, loss_value = sess.run([gd_op, loss])

                if step % 100 == 0:
                    print('At step {} loss is {:.02f}'.format(step, loss_value))
                    saver.save(sess, save_path='./output_mnist/mnist')
                step += 1

        except tf.errors.OutOfRangeError:
            print('Training ended')
            saver.save(sess, save_path='./output_mnist/mnist')


# Try to load the model and feed it with a constant

def load_and_run(n):
    with tf.Graph().as_default(), tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        dummy_input = tf.constant(np.ones((n, IMAGE_SIZE ** 2)), dtype=tf.float32)
        importer = tf.train.import_meta_graph('./output_mnist/mnist.meta', import_scope='mnist',
                                              input_map={'mnist_input': dummy_input})
        output = tf.get_collection('output')[0]

        importer.restore(sess, './output_mnist/mnist')
        sess.run(init_op)

        return sess.run([output])


def main():
    # train()
    print(load_and_run(3))


if __name__ == '__main__':
    main()
