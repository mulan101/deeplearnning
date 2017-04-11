'''
Created on 2017. 3. 21.

@author: 한제호
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
dropout_rate = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# 이미지당 784개의 열을 28X28 형태로 변경해줌
X_img = tf.reshape(X, [-1, 28, 28, 1])

# input값 X_img : (28,28,1)
# 3X3형태의 필터를 32개 생성
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
# padding방식을 사용하고 필터이동은 1X1형태로 이동함
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
# 2X2형태의 커널을 사용하고 커널이동은 2X2형태로 이동함 
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
'''
결과값
conv2d : (?,28,28,32)
relu : (?,28,28,32)
max_pool : (?,14,14,32)
'''

# input값  L1 : (?,14,14,32)
# 3X3형태의 필터를 64개 생성
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
# padding방식을 사용하고 필터이동은 1X1형태로 이동함
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
# 2X2형태의 커널을 사용하고 커널이동은 2X2형태로 이동함 
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#Fully-connected layer 진입전 이미지당 1X3136형태로 펼처줌
L2 = tf.reshape(L2, [-1, 7 * 7 * 64])
'''
conv2d : (?, 14,14,64)
relu : (?, 14,14,64)
max_pool : (?, 7,7,64)
reshape : (?, 3136)
'''

# input 3136 output : 10
W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3) + b

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
'''

Epoch: 0001 cost = 0.380970516
Epoch: 0002 cost = 0.097521901
Epoch: 0003 cost = 0.069546332
Epoch: 0004 cost = 0.055918751
Epoch: 0005 cost = 0.046638948
Epoch: 0006 cost = 0.040767722
Epoch: 0007 cost = 0.035574222
Epoch: 0008 cost = 0.031549309
Epoch: 0009 cost = 0.027653358
Epoch: 0010 cost = 0.024728205
Epoch: 0011 cost = 0.021958173
Epoch: 0012 cost = 0.018924782
Epoch: 0013 cost = 0.017573998
Epoch: 0014 cost = 0.015098593
Epoch: 0015 cost = 0.013729086
Learning Finished!
Accuracy: 0.9882
'''