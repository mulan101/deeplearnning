'''
Created on 2017. 3. 15.
'''

import tensorflow as tf

# 더하기
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print("node1: ", node1, "node2: ", node2, "node3: ",node3)

with tf.Session() as sess:
    print("sess.run(node1, node2): " , sess.run([node1, node2]))
    print("sess.run(node3): ", sess.run(node3))