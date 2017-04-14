'''
Created on 2017. 3. 21.

@author: 한제호
'''
from tensorflow.examples.tutorials.mnist import input_data
from prototype.cnn.cnn_model import Model
import numpy as np
import tensorflow as tf

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    models = [] 
    num_models = 5 
    
    for m in range(num_models): 
        models.append(Model(sess, "model" + str(m), learning_rate)) 
 
    sess.run(tf.global_variables_initializer()) 
    
    print('Learning Started!') 

 
    for epoch in range(training_epochs): 
        avg_cost_list = np.zeros(len(models)) 
        total_batch = int(mnist.train.num_examples / batch_size) 
        for i in range(total_batch): 
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) 
            # train each model 
            for m_idx, m in enumerate(models): 
                c, _ = m.train(batch_xs, batch_ys) 
                avg_cost_list[m_idx] += c / total_batch 
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list) 
        
    print('Learning Finished!') 
 
 
    # Test model and check accuracy 
    test_size = len(mnist.test.labels) 
    predictions = np.zeros(test_size * 10).reshape(test_size, 10) 
    for m_idx, m in enumerate(models): 
        print(m_idx, 'Accuracy:', m.get_accuracy(mnist.test.images, mnist.test.labels)) 
        p = m.predict(mnist.test.images) 
        predictions += p 
    
    ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1)) 
    ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32)) 
    print('Ensemble accuracy:', sess.run(ensemble_accuracy)) 

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