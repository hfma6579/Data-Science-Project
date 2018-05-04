"""
CNN with 4 convolution layer and 1 dense layer
Achieve 0.99228 in MNIST data recognition
"""
import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


NUM_EPOCHS = 5           # Patience in early stopping
BATCH_SIZE = 64          # Batch size
CROSS_EVAL = True        # Mode for cross evaluation


def train_input_fn(features, labels, batch_size):
    # iterator for training data
    dataset = tf.data.Dataset.from_tensor_slices(
        ({'pixel': features}, labels))
    dataset = dataset.shuffle(BATCH_SIZE).repeat().batch(batch_size)
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
    input_layer = tf.reshape(tf.feature_column.input_layer(
        features, params['feature_columns']), [-1, 28, 28, 1])
    input_layer = tf.layers.dropout(inputs=input_layer, rate=0.1)

    regularizer = tf.contrib.layers.l2_regularizer(0.1, scope=None)

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        strides=(1, 1),
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_regularizer=regularizer
    )

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        strides=(2, 2),
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_regularizer=regularizer
    )

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        strides=(1, 1),
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_regularizer=regularizer
    )

    # Convolutional Layer #4
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        strides=(2, 2),
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_regularizer=regularizer
    )

    # Dense Layer
    conv4_flat = tf.reshape(conv4, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(
        inputs=conv4_flat,
        units=1024,
        activation=tf.nn.relu,
        kernel_regularizer=regularizer
    )
    dense = tf.layers.dropout(inputs=dense)

    # Logits Layer
    logits = tf.layers.dense(inputs=dense, units=10,
                             kernel_regularizer=regularizer)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"classes": tf.argmax(input=logits, axis=1)})

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits
    )

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=0.001,
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
    # Load training data
    data = np.load('data/train.npz')

    # Create feature columns, here there is only one feature (* 28 * 28).
    feature_columns = [
        tf.feature_column.numeric_column(
            key="pixel",
            shape=[28, 28],
            normalizer_fn=lambda x: tf.image.per_image_standardization(x))]

    # create classifier
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        params={
            'feature_columns': feature_columns
        })

    if CROSS_EVAL:
        train_x, eval_x, train_y, eval_y = train_test_split(
            data['x'], data['y'].astype(np.int32), test_size=0.2, random_state=0)

        # performs a early stopping for each $NUM_EPOCHS epoches
        steps_per_train = train_y.size * NUM_EPOCHS // BATCH_SIZE
        old_eval_result = 0
        counter = 0

        while(counter < 3):
            counter += 1
            classifier.train(
                input_fn=lambda: train_input_fn(train_x, train_y, BATCH_SIZE),
                steps=steps_per_train)

            new_eval_result = classifier.evaluate(
                input_fn=lambda: eval_input_fn(eval_x, eval_y, BATCH_SIZE))['accuracy']

            print("Accuracy: {}".format(new_eval_result))
            # if there is a improve on the evaluation set, set the counter to 0
            if new_eval_result - old_eval_result > 1E-5:
                counter = 0
            old_eval_result = new_eval_result
            print("Current counter: {}".format(counter))
            return

    full_x, full_y = data['x'], data['y'].astype(np.int32)
    classifier.train(
        input_fn=lambda: train_input_fn(full_x, full_y, BATCH_SIZE),
        steps=30000
    )
    test_x = np.load('data/test.npz')['x']
    predictions = classifier.predict(
        input_fn=lambda: predict_input_fn(test_x, BATCH_SIZE))
    labels = [pred['classes'] for pred in predictions]
    df = pd.DataFrame({'ImageId': range(1, len(labels) + 1), 'Label': labels})
    df.set_index('ImageId', inplace=True)
    df.to_csv('submission.csv')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
