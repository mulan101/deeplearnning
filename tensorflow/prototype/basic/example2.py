'''
Created on 2017. 3. 15.
'''

import tensorflow as tf

#입력받아처리
a = tf.placeholder(tf.float32, name='a')
b = tf.placeholder(tf.float32, name='b')
with tf.name_scope("adder_node") as scope: 
    adder_node = a + b
    
with tf.Session() as sess:
    #writer = tf.summary.FileWriter('C:/Python/tensorboard_log',sess.graph)
    print(sess.run(adder_node, feed_dict={a:3, b:4}))
    print(sess.run(adder_node, feed_dict={a:[1,2], b:[3,4]}))