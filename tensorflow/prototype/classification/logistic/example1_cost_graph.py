'''
Created on 2017. 3. 17.

@author: 한제호
'''
import tensorflow as tf
import matplotlib.pyplot as plt

x_train = [1, 2, 3, 4]
y_train = [0, 0, 1, 1]

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

W = tf.placeholder(tf.float32, name='W')

hypothesis = tf.sigmoid(X * W)

#cost = tf.reduce_mean(tf.square(hypothesis - Y))
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    W_val = []
    cost_val = []
    for i in range(-30, 50):
        feed_W = i * 0.1
        curr_cost, curr_W = sess.run([cost, W], feed_dict={X: x_train, Y: y_train, W: feed_W})
        W_val.append(curr_W)
        cost_val.append(curr_cost)

    # Show the cost function
    plt.plot(W_val, cost_val)
    plt.show()