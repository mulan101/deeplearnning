'''
Created on 2017. 3. 17.

@author: 한제호
'''
import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name='weight')

hypothesis = X * W

cost = tf.reduce_sum(tf.square(hypothesis - Y))

learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(21):  
        _, cost_val, W_val = sess.run([update, cost, W], feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, W_val)