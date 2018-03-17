import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt
import numpy as np

###########################################
# Basic 2 steps of any TF model :-        # 
#                                         #
# 1. Create the computation Graph         #
# 2. Executing the computation graph      #
#                                         # 
###########################################


# Read the data(Done) 
# Here we will be reading the input data
features_data, prices_data = load_boston(True)
# Training data
features_data = scale(features_data)
prices_data = prices_data


# Here we need to first create a way to handle input using placeholders from tensorflow 
# Input layer
x = tf.placeholder(tf.float64, shape = (506,13))
y = tf.placeholder(tf.float64, shape = (506))

# Then we need to write the equations for as many hidden layers as we require
# Hidden Layer 1(with 20 Neurons)
w1_h = tf.Variable(tf.zeros([13,20], dtype = tf.float64))
b1_h = tf.Variable(tf.zeros([20], dtype = tf.float64))
x1_output = tf.add(tf.matmul(x,w1_h),b1_h)

# Hidden Layer 2(With 15 Neurons)
w2_h = tf.Variable(tf.zeros([20,15], dtype = tf.float64))
b2_h = tf.Variable(tf.zeros([15], dtype = tf.float64))
x2_output = tf.add(tf.matmul(x1_output,w2_h),b2_h)

# Hidden Layer 3(With 10 Neurons)
w3_h = tf.Variable(tf.zeros([15,10], dtype = tf.float64))
b3_h = tf.Variable(tf.zeros([10], dtype = tf.float64))
x3_output = tf.add(tf.matmul(x2_output,w3_h), b3_h)

# Output Layer
# Here we have just 1 Neuron in the output layer
w_out = tf.Variable(tf.zeros([10,1], dtype = tf.float64 ))
b_out = tf.Variable(tf.zeros([1], dtype = tf.float64))
y_pred  = tf.add(tf.matmul(x3_output,w_out),b_out)


# Then we need to need to find the cost function(We use MSE)
cost = tf.reduce_mean(tf.square(y_pred -y))

# Information in regard to the plan of training the model
epochs = 3000  # This is in regard to the number of times the training has to be done
error_results = [[],[]] # Here we insert the errors
learning_rate = 0.025

# Optimization of weights                           
# On the bases of the cost and learning rate we fine tune the weights
optimizer =  tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# Initializing the Global Variables
init = tf.global_variables_initializer()


# Then finally we train our model(Creating the exution step of the above graph)
with tf.Session() as sess:
     sess.run(init)
     for i in list(range(3000)):
          sess.run(optimizer, feed_dict = {x: features_data, y:prices_data})
          if(i%10 == 0):
             error_results[0].append(i+1)
             error_results[1].append(sess.run(cost, feed_dict = {x: features_data, y: prices_data}))
          
          if(i%100 == 0):
             print(sess.run(cost, feed_dict = {x: features_data, y:prices_data})) 
           
     plt.plot(error_results[0], error_results[1], 'r--')
     plt.axis([0, epochs, 50, 600])
     plt.show()

