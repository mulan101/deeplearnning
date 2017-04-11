'''
Created on 2017. 4. 11.

@author: 한제호
'''
import tensorflow as tf
import numpy as np

#(1,2,2,1)
image = np.array([[[[4],[3]],[[2],[1]]]], dtype=np.float32)
sess = tf.InteractiveSession()
pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

print(pool.shape)
print(pool.eval())