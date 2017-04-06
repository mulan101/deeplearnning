'''
Created on 2017. 3. 21.

@author: 한제호
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# weights & bias for nn layers
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# default
'''
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
'''
#softmax_cross_entropy_with_logits and AdamOptimizer
hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
    
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch
    
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')
    
    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={
          X: mnist.test.images, Y: mnist.test.labels}))


''' default
Epoch: 0001 cost = 2.046914017
Epoch: 0002 cost = 1.655541348
Epoch: 0003 cost = 1.395275627
Epoch: 0004 cost = 1.217845253
Epoch: 0005 cost = 1.091956704
Epoch: 0006 cost = 0.998918433
Epoch: 0007 cost = 0.927551316
Epoch: 0008 cost = 0.871126612
Epoch: 0009 cost = 0.825361123
Epoch: 0010 cost = 0.787444251
Epoch: 0011 cost = 0.755466313
Epoch: 0012 cost = 0.728108003
Epoch: 0013 cost = 0.704385954
Epoch: 0014 cost = 0.683610103
Epoch: 0015 cost = 0.665233025
Learning Finished!
Accuracy: 0.8662
'''

''' softmax_cross_entropy_with_logits and AdamOptimizer
Epoch: 0001 cost = 0.646840433
Epoch: 0002 cost = 0.357035918
Epoch: 0003 cost = 0.316992506
Epoch: 0004 cost = 0.297901344
Epoch: 0005 cost = 0.286603179
Epoch: 0006 cost = 0.278914297
Epoch: 0007 cost = 0.273144604
Epoch: 0008 cost = 0.268446943
Epoch: 0009 cost = 0.265257705
Epoch: 0010 cost = 0.261988746
Epoch: 0011 cost = 0.259241566
Epoch: 0012 cost = 0.256946926
Epoch: 0013 cost = 0.255295035
Epoch: 0014 cost = 0.253231567
Epoch: 0015 cost = 0.251651690
Learning Finished!
Accuracy: 0.9277
'''