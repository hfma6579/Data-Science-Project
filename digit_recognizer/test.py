import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split

NUM_EPOCHS = 5
BATCH_SIZE = 64


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(
        ({'pixel': features}, labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset


def eval_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(
        ({'pixel': features}, labels))
    dataset = dataset.batch(batch_size)
    return dataset


def cnn_model_fn(features, labels, mode, params):
    # input layer
    input_layer = tf.reshape(tf.feature_column.input_layer(
        features, params['feature_columns']), [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        strides=(1, 1),
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
    )

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        strides=(2, 2),
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        strides=(1, 1),
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
    )

    # Convolutional Layer #4
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        strides=(2, 2),
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    # Dense Layer
    conv4_flat = tf.reshape(conv4, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(
        inputs=conv4_flat,
        units=1024,
        activation=tf.nn.relu
    )

    # Logits Layer
    logits = tf.layers.dense(inputs=dense, units=10)

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits
    )

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
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
    train_x, eval_x, train_y, eval_y = train_test_split(
        data['x'], data['y'].astype(np.int32), random_state=0)

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
            'feature_columns': feature_columns,
        })

    # performs a early stopping for each $NUM_EPOCHS epoches
    steps_per_train = train_y.size * NUM_EPOCHS // BATCH_SIZE
    old_eval_result = 0
    counter = 0

    while(counter < 2):
        counter += 1
        classifier.train(
            input_fn=lambda: train_input_fn(train_x, train_y, BATCH_SIZE),
            steps=steps_per_train)

        new_eval_result = classifier.evaluate(
            input_fn=lambda: eval_input_fn(eval_x, eval_y, BATCH_SIZE))['accuracy']

        print(new_eval_result)
        # if there is a improve on the evaluation set, set the counter to 0
        if new_eval_result > old_eval_result:
            counter = 0


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
