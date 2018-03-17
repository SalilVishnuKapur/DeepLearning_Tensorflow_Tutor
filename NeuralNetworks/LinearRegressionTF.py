import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt
import numpy as np

# Read the data 
# Here we will be reading the input data
features_data, prices_data = load_boston(True)
# Training data
features_data = scale(features_data)
prices_data = prices_data
'''
# Validation data
validation_features = scale(features_data[300:400])
validation_prices = prices_data[300:400]
# Testing data
test_features = scale(features_data[400:])
test_prices = prices_data[400:]
'''

# Now we will take here the input placeholders
x = tf.placeholder(tf.float64, shape = (506, 13))
y = tf.placeholder(tf.float64, shape = (506,))

# Formulate the equation y = wx + b
# We need to hold the input x values into a placeholder but we will keep it for # later
# First of all we declare a weight matrix
w = tf.Variable(tf.truncated_normal([13,1], mean=0.0, stddev = 1.0, dtype = tf.float64))
# Then a bias matrix is declared
b = tf.Variable(tf.zeros(1, dtype = tf.float64))

'''
def computation(x, actual_y):
   y = tf.add(tf.matmul(x,w), b)
   y_pred = sess.run(y, feed_dict = {x: features_data})
   cost = tf.reduce_mean(tf.square(y - actual_y))
   return(y, cost)
'''

# Here we are going to declare some global variables
epochs = 3000
learning_rate = 0.025
points = [[], []]

# Train computation
#y_pred, cost = computation(train_features, train_prices)

# Here we will declare few values that will optimize the cost value and get pred
# Now this is not correct as you have to think why we thought of using this API 
# tensorflow. This is a great library and it creates the graphs and then runs the
init = tf.global_variables_initializer()
#optimizer =  tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

y_pred = tf.add(b, tf.matmul(x,w))
cost = tf.reduce_mean(tf.square(y_pred - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Then start the session
with tf.Session() as sess:
    sess.run(init)
    for i in list(range(epochs)):
         sess.run(optimizer, feed_dict = {x: features_data, y : prices_data})
         if(i%10 == 0):
           points[0].append(i+1)
           points[1].append(sess.run(cost, feed_dict = {x: features_data, y : prices_data}))
          
         if(i%100 == 0):
           print(sess.run(cost, feed_dict = {x: features_data, y : prices_data }))
    plt.plot(points[0], points[1], 'r--')
    plt.axis([0, epochs, 50, 600])
    plt.show()
    # Validation error calculation
    #valid_cost = computation(validation_features, validation_prices)[1]
    #print('Validation error =', sess.run(valid_cost))
    
    # Test error calculation 
    #test_cost = computation(test_features, test_prices)
    #print('Test error =',sess.run(test_cost))
