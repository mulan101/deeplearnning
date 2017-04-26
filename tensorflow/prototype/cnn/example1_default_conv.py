'''
Created on 2017. 4. 11.

@author: 한제호
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#(1,3,3,1)
image = np.array([[[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]]], dtype=np.float32)

sess = tf.InteractiveSession()
print('image shape', image.shape)
#print('image reshape', image.reshape(3,3))
#plt.imshow(image.reshape(3,3), cmap='Greys')
#plt.show()

#필터 (2,2,1,3)
weight = tf.constant([[[[1., 10., -1.]],[[1., 10., -1.]]],[[[1., 10., -1.]],[[1., 10., -1.]]]])
print('weight.shape', weight.shape)
#conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='VALID')
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')

#Tensor 실행
conv2d_img = conv2d.eval()
print('conv2d_img.shape', conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img, 0 , 3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,3,i+1)
    plt.imshow(one_img.reshape(3,3), cmap='gray')
    plt.show()
 