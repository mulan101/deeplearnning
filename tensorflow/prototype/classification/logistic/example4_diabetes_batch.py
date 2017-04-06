'''
Created on 2017. 4. 3.

@author: 한제호
'''
import tensorflow as tf

filename_queue = tf.train.string_input_producer(['data-03-diabetes.csv'], shuffle=False, name='filename_queue')
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defults)

train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=100)

X = tf.placeholder(tf.float32, shape=[None, 8], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')

W = tf.Variable(tf.random_normal(shape=[8, 1]), name='weight')
b = tf.Variable(tf.random_normal(shape=[1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    for step in range(2001):
        x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_batch, Y: y_batch})
        if step % 200 == 0:
            print(step, cost_val)
    
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_batch, Y: y_batch})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
    
    coord.request_stop()
    coord.join(threads)
   