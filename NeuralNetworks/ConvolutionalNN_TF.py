import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#################################################
# The main two purposes of Tensorflow are :-    #
# 1. To create the tensorflow graph             #   
# 2. To run the computation of tensorflow graph #
#################################################


# Read the data from MNIST
# This reading process does not involve tensorflow
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])


# (Convolutional_layer + pooling) => 1
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], dtype = tf.float32))
b_conv1 = tf.Variable(tf.zeros([32], dtype= tf.float32))
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides = [1,1,1,1], padding = 'SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


# (Convolutional_layer + pooling) => 2
W_conv2 = tf.Variable(tf.truncated_normal([5,5,32, 64], dtype = tf.float32))
b_conv2 = tf.Variable(tf.zeros([64], dtype = tf.float32))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides = [1,1,1,1], padding = 'SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


# (Fully Connected layer(1024 Neurons)) => 1
W_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], dtype = tf.float32))
b_fc1 = tf.Variable(tf.zeros([1024], dtype = tf.float32))
h_pool1_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)


# (Fully Connected layer(10 Neurons)) => 2
w_out = tf.Variable(tf.truncated_normal([1024,10], dtype = tf.float32))
b_out = tf.Variable(tf.zeros([10], dtype = tf.float32))
y_out = tf.matmul(h_fc1, w_out) + b_out 


# Find cost and optimizing the weights
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_out))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Running the session here
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2200):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    for i in range(2200):
        batch = mnist.test.next_batch(50)
        if i % 100 == 0:
            test_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
            print('test accuracy %g' % test_accuracy)

