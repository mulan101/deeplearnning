'''
Created on 2017. 3. 21.

@author: 한제호
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from prototype.cnn.cnn_model import Model
tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    model = Model(sess, "model", learning_rate)
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
    
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = model.train(batch_xs, batch_ys)
            avg_cost += c / total_batch
    
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')
    
    # Test model and check accuracy
    print('Accuracy:', model.get_accuracy(mnist.test.images, mnist.test.labels))
'''
Learning Started!
Epoch: 0001 cost = [ 0.40938698  0.45928397]
Epoch: 0002 cost = [ 0.09741164  0.10563979]
Epoch: 0003 cost = [ 0.07519913  0.07593221]
Epoch: 0004 cost = [ 0.06030078  0.06108334]
Epoch: 0005 cost = [ 0.05306817  0.05424173]
Epoch: 0006 cost = [ 0.04682166  0.04844619]
Epoch: 0007 cost = [ 0.04236656  0.04129622]
Epoch: 0008 cost = [ 0.039807    0.03906083]
Epoch: 0009 cost = [ 0.0361566   0.03600198]
Epoch: 0010 cost = [ 0.03231378  0.03577451]
Epoch: 0011 cost = [ 0.03110797  0.0326055 ]
Epoch: 0012 cost = [ 0.0288111   0.03042327]
Epoch: 0013 cost = [ 0.0291169  0.0289884]
Epoch: 0014 cost = [ 0.02816853  0.0264328 ]
Epoch: 0015 cost = [ 0.02781756  0.02613632]
Learning Finished!
0 Accuracy: 0.994
1 Accuracy: 0.9917
Ensemble accuracy: 0.9931
'''