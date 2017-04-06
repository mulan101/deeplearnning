'''
Created on 2017. 3. 20.

@author: 한제호

real데이터와  test데이터 분리 sample
'''
import tensorflow as tf

#학습데이터
x_data = [[1, 2, 1], [1, 3, 2], [1, 3, 4], [1, 5, 5], [1, 7, 5], [1, 2, 5], [1, 6, 6], [1, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

#테스트데이터
x_test = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]
y_test = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
nb_classes = 3

X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.int32, [None, 3])

W = tf.Variable(tf.random_normal([3, nb_classes]), name="weight")
b = tf.Variable(tf.random_normal([nb_classes]), name="bias")

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits =logits))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#예측값의 Max값 추출
prediction = tf.argmax(hypothesis, 1)
#실제값과 예측값의 일치여부(True/False)
correct_prediction = tf.equal(prediction, tf.argmax(Y, 1))
#실제값과의 일치 평균
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        sess.run(optimizer, feed_dict = {X: x_data, Y: y_data})
        if step % 100 == 0:
            cost_val, acc = sess.run([cost,accuracy], feed_dict={X: x_data, Y: y_data})
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc))
 
    pre_val, correct_val = sess.run([prediction, correct_prediction], feed_dict = {X: x_test, Y: y_test})
    for x, p, c in zip(x_test, pre_val, correct_val):
        print(x, p, c)
    