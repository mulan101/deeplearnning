'''
Created on 2017. 4. 3.

@author: 한제호
'''
import tensorflow as tf

tf.set_random_seed(777)

filename_queue = tf.train.string_input_producer(['data-01-test-score.csv'], shuffle=False, name='filename_queue')
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defults)

train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=6)


X = tf.placeholder(tf.float32, shape=[None, 3], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
tf.summary.histogram("weight", W)

with tf.name_scope('hypothesis') as scope:
    hypothesis = tf.matmul(X, W) + b
with tf.name_scope('cost') as scope:
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    tf.summary.scalar('cost', cost)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('C:/Python/tensorboard_log',sess.graph)
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    for step in range(2001):
        x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
        cost_val, hy_val, _, summary = sess.run([cost, hypothesis, train, merged], feed_dict={X: x_batch, Y: y_batch})
        writer.add_summary(summary, step)
        if step % 10 == 0:
            print(step, "Cost: ", cost_val, "Prediction: ", hy_val)
    
    coord.request_stop()
    coord.join(threads)

    print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))
    print("Other scores will be ", sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))