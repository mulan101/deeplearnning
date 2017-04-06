'''
Created on 2017. 4. 3.

@author: 한제호
'''
import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7

X = tf.placeholder(tf.float32, shape=[None, 16], name='X')
Y = tf.placeholder(tf.int32, shape=[None, 1], name='Y')
# one hot 형태로 변경
Y_one_hot = tf.one_hot(Y, nb_classes)
# 3차원에서 2차원으로 변경
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal(shape=[16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal(shape=[nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_one_hot, logits=logits))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

#예측값의 Max값 추출
prediction = tf.argmax(hypothesis, 1)
#실제값과 예측값의 일치여부(True/False)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
#실제값과의 일치 평균
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        sess.run([optimizer], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            cost_val, acc = sess.run([cost,accuracy], feed_dict={X: x_data, Y: y_data})
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc))
            
    print(sess.run(prediction, feed_dict={X:[[1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1]]}))        
            
