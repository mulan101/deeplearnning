'''
Created on 2017. 4. 11.

@author: 한제호
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
img = mnist.train.images[10].reshape(28, 28)
#plt.imshow(img, cmap='gray')
#plt.show()

sess = tf.InteractiveSession()
img = img.reshape(-1, 28, 28, 1)
#필터 (3,3,1,5)
weight = tf.Variable(tf.random_normal([3,3,1,5], stddev=0.01))
conv2d = tf.nn.conv2d(img, weight, strides=[1, 2, 2, 1], padding='SAME')
print('conv2d', conv2d)

pool = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print('pooling', pool)

sess.run(tf.global_variables_initializer())
pool_img = pool.eval()
pool_img = np.swapaxes(pool_img, 0 , 3)
for i, one_img in enumerate(pool_img):
    print(one_img.reshape(7,7))
    plt.subplot(1,5,i+1)
    plt.imshow(one_img.reshape(7,7), cmap='gray')
    plt.show()