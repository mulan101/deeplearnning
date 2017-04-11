'''
Created on 2017. 4. 10.

@author: 한제호
'''
import tensorflow as tf
import os
from PIL import Image

image_dir = "C:/Python/image/Test1.jpg"

filename_list = [image_dir]
filename_queue = tf.train.string_input_producer(filename_list)

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
image_decoded = tf.image.decode_jpeg(value)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)
    image = sess.run(image_decoded)
    print(image)
    Image.fromarray(image).show()
    
    coord.request_stop()
    coord.join(thread)
    