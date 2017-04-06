'''
Created on 2017. 3. 29.

@author: 한제호
'''
import tensorflow as tf

# X,Y 데이터 선언
x_train = [1, 2, 3]
y_train = [1, 2, 3]

#W와 b 1차원 변수 선언
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#가설
with tf.name_scope('hyponthesis') as scope:
    hyponthesis = x_train * W + b

#비용
with tf.name_scope('cost') as scope:
    cost = tf.reduce_mean(tf.square(hyponthesis - y_train))
    tf.summary.scalar('cost', cost)

#cost값이 최소화가 되도록 선언
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('C:/Python/tensorboard_log',sess.graph)
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, cost_val, W_val, b_val, summary = sess.run([train, cost, W, b, merged])
        writer.add_summary(summary, step)
        if step % 200 == 0:
            print('STEP : {}, COST : {}, W : {}, b : {}'.format(step, cost_val, W_val, b_val))
    