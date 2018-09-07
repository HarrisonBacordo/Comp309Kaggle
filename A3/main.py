# import tensorflow as tf
# import numpy as np
#
# INPUT_SIZE = 17
#
# tf.logging.set_verbosity(tf.logging.INFO)
#
#
# def model(features, labels, mode):
#     input_layer = tf.reshape(features["x"], [-1, INPUT_SIZE])
#
#     hl1 = tf.layers.dense(inputs=input_layer, units=15, activation=tf.nn.relu)
#
#     drop1 = tf.layers.dropout(inputs=hl1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
#
#     # hl2 = tf.layers.dense(inputs=drop1, units=15, activation=tf.nn.relu)
#     #
#     # drop2 = tf.layers.dropout(inputs=hl2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
#
#     logits = tf.layers.dense(inputs=drop1, units=14)
#
#     epsilon = tf.constant(1e-8)
#     logits = logits + epsilon
#
#     predictions = {
#         "classes": tf.argmax(input=logits, axis=1),
#         "probabilities": tf.nn.softmax(logits=logits, name="softmax_tensor")
#     }
#     if mode == tf.estimator.ModeKeys.PREDICT:
#         return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
#
#     loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
#
#     if mode == tf.estimator.ModeKeys.TRAIN:
#         optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
#         train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
#         return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
#
#     eval_metric_ops = {
#         "accuracy": tf.metrics.accuracy(
#             labels=labels, predictions=predictions["classes"])
#     }
#     return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
#
#
# def main(unused_argv):
#     train_data = np.genfromtxt("train_data.csv", delimiter=",", dtype=np.float32)
#     test_data = np.genfromtxt("test_data.csv", delimiter=",", dtype=np.float32)
#     train_labels = np.genfromtxt("train_labels.csv", delimiter=",", dtype=np.int32)
#     test_labels = np.genfromtxt("test_labels.csv", delimiter=",", dtype=np.int32)
#
#     classif_model = tf.estimator.Estimator(model_fn=model)
#
#     # Set up logging for predictions
#     # Log the values in the "Softmax" tensor with label "probabilities"
#     tensors_to_log = {"probabilities": "softmax_tensor"}
#     logging_hook = tf.train.LoggingTensorHook(
#         tensors=tensors_to_log, every_n_iter=50)
#
#     # Train the model
#     train_input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={"x": train_data},
#         y=train_labels,
#         batch_size=10,
#         num_epochs=None,
#         shuffle=False)
#     classif_model.train(
#         input_fn=train_input_fn,
#         steps=5000,
#         hooks=[logging_hook])
#
#     # Evaluate the model and print results
#     eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={"x": test_data},
#         y=test_labels,
#         num_epochs=1,
#         shuffle=False)
#     eval_results = classif_model.evaluate(input_fn=eval_input_fn)
#     print(eval_results)
#
#
# if __name__ == '__main__':
#     tf.app.run()


#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 17])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=input_layer, units=20, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00000001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    train_data = np.genfromtxt("train_data.csv", delimiter=",", dtype=np.float32)
    test_data = np.genfromtxt("test_data.csv", delimiter=",", dtype=np.float32)
    train_labels = np.genfromtxt("train_labels.csv", delimiter=",", dtype=np.int32)
    test_labels = np.genfromtxt("test_labels.csv", delimiter=",", dtype=np.int32)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
