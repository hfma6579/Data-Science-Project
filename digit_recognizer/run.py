"""
CNN with 2 Convolution layers for MNIST Data set

"""
import os

import tensorflow as tf
import numpy as np
import pandas as pd

# flag settings
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_dir', './tmp/mnist_cnn', """Path to the model directory.""")
tf.app.flags.DEFINE_string('data_dir', './data/', """Path to the data directory.""")
tf.app.flags.DEFINE_bool('heldout', False, """Perform cross evluation.""")
tf.app.flags.DEFINE_bool('predict', False, """Perform prediction.""")
tf.app.flags.DEFINE_integer('max_iter', 10000, """Maximum number of iteration""")

BATCH_SIZE = 64
EVAL_BATCH_SIZE = 1024


def train_input_fn(features, labels, batch_size):
    # iterator for training data
    dataset = tf.data.Dataset.from_tensor_slices(
        ({'pixel': features}, labels))
    dataset = dataset.shuffle(batch_size).repeat().batch(batch_size)
    return dataset


def eval_input_fn(features, labels, batch_size):
    # iterator for evaluation data
    dataset = tf.data.Dataset.from_tensor_slices(
        ({'pixel': features}, labels))
    dataset = dataset.batch(batch_size)
    return dataset


def predict_input_fn(features, batch_size):
    # iterator for prediction data
    dataset = tf.data.Dataset.from_tensor_slices(
        {'pixel': features})
    dataset = dataset.batch(batch_size)
    return dataset


def cnn_model_fn(features, labels, mode, params):
    # input layer
    input_layer = tf.feature_column.input_layer(features, params['feature_columns'])
    input_layer = tf.reshape(input_layer, [-1, 28, 28, 1])
    tf.summary.image('input', input_layer, 1)

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        strides=(2, 2),
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name='CONV1'
    )

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        strides=(2, 2),
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name='CONV2'
    )

    # Dense Layer
    conv2_flat = tf.reshape(conv2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(
        inputs=conv2_flat,
        units=1024,
        activation=tf.nn.relu,
        name='DENSE'
    )
    dense = tf.layers.dropout(dense, training=mode==tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(
        inputs=dense, 
        units=10,
        name='LOGIT'
    )

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"classes": tf.argmax(input=logits, axis=1)})

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits
    )

    if mode == tf.estimator.ModeKeys.TRAIN:

        # create feature map plot
        with tf.name_scope("feature_map"):
            tf.summary.image('conv1', tf.reshape(conv1, [-1, 14, 14, 1]), max_outputs=8)
            tf.summary.image('conv2', tf.reshape(conv2, [-1, 7, 7, 1]), max_outputs=8)
        
        # adam optimizer performs well with drop-out layer
        optimizer = tf.train.AdamOptimizer(
            learning_rate=0.001
        )

        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    else:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels,
                predictions=tf.argmax(input=logits, axis=1)
            )
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(argv):
    feature_columns = [
        tf.feature_column.numeric_column(
            key="pixel",
            shape=[28, 28],
            normalizer_fn=lambda x: tf.image.per_image_standardization(x)
        )
    ]

    estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=FLAGS.model_dir,
        params={
            'feature_columns': feature_columns
        }
    )

    if FLAGS.predict:
        # load text data and create submission csv
        test_x = np.load(os.path.join(FLAGS.data_dir, 'test.npz'))['x']

        predictions = estimator.predict(
            input_fn=lambda: predict_input_fn(test_x, BATCH_SIZE)
        )

        labels = [pred['classes'] for pred in predictions]
        df = pd.DataFrame({'ImageId': range(1, len(labels) + 1), 'Label': labels})
        df.set_index('ImageId', inplace=True)
        df.to_csv('submission.csv')

    elif FLAGS.heldout:
        # perform held-out evaluation, check tensorboard for visualization
        data = np.load(os.path.join(FLAGS.data_dir, 'train.npz'))

        # extract a sample data (with EVAL_BATCH_SIZE) for evalution
        x, y = data['x'], data['y'].astype(int)
        eval_ind = np.random.choice(np.arange(y.size), EVAL_BATCH_SIZE, replace=False)
        train_ind = np.delete(np.arange(y.size), eval_ind)
        train_x, train_y = x[train_ind], y[train_ind]
        eval_x, eval_y = x[eval_ind], y[eval_ind]

        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: train_input_fn(train_x, train_y, BATCH_SIZE), max_steps=FLAGS.max_iter
        )

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: eval_input_fn(eval_x, eval_y, EVAL_BATCH_SIZE),
            start_delay_secs=60,
            throttle_secs=60
        )

        tf.estimator.train_and_evaluate(
            estimator=estimator,
            train_spec=train_spec,
            eval_spec=eval_spec
        )
    
    else:
        # load entire training set for training (default mode)
        data = np.load(os.path.join(FLAGS.data_dir, 'train.npz'))

        train_x, train_y = data['x'], data['y'].astype(int)

        estimator.train(
            input_fn=lambda: train_input_fn(train_x, train_y, BATCH_SIZE), 
            max_steps=FLAGS.max_iter
        )



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
