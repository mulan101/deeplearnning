'''
Created on 2017. 4. 3.

@author: 한제호
'''
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3], name='input')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='output')

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
tf.summary.histogram("weight", W)

# Hypothesis
with tf.name_scope('hypothesis') as scope:
    hypothesis = tf.matmul(X, W) + b
# Simplified cost/loss function
with tf.name_scope('cost') as scope:
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    tf.summary.scalar('cost', cost)
# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('C:/Python/tensorboard_log',sess.graph)
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _, summary = sess.run([cost, hypothesis, train, merged], feed_dict={X: x_data, Y: y_data})
    writer.add_summary(summary, step)
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "Prediction: ", hy_val)


# Ask my score
print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))
print("Other scores will be ", sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))