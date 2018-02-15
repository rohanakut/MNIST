# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 01:37:04 2017
@author: Rohan-PC
"""

import tensorflow as tf
import numpy as np
import time
#import keras
#keep_prob = tf.placeholder("float",None)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
def hidden_layer_ini(in_size,out_size):
    hidden = tf.Variable(tf.truncated_normal([in_size,out_size],stddev=0.1))
    return hidden
def output_layer_ini(in_size,out_size):
    output = tf.Variable(tf.truncated_normal([in_size,out_size],stddev = 0.1))
    return output
def for_prop(x,w1,b1,w2,b2,w3,b3,w4,b4,w5,b5):
    activation1 = (tf.nn.relu(tf.matmul(x,w1)+b1))
    activation2 = (tf.nn.relu(tf.matmul(activation1,w2)+b2))
    activation3 = (tf.nn.relu(tf.matmul(activation2,w4)+b4))
    activation4 = tf.nn.dropout(tf.nn.relu(tf.matmul(activation3,w5)+b5),0.5)
    output = (tf.matmul(activation4,w3)+b3)
    return output
regu1 = 0.001
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
weight1 = hidden_layer_ini(784,400)
weight2 = hidden_layer_ini(400,200)
weight4 = hidden_layer_ini(200,100)
weight5 = hidden_layer_ini(100,40)
weight3 = output_layer_ini(40,10)
bias1 = tf.Variable(tf.zeros([weight1.shape[1]]))
bias2 = tf.Variable(tf.zeros([weight2.shape[1]]))
bias4 = tf.Variable(tf.zeros([weight4.shape[1]]))
bias5 = tf.Variable(tf.zeros([weight5.shape[1]]))
bias3 = tf.Variable(tf.zeros([weight3.shape[1]]))
''''
forwardpropagation
'''
output = for_prop(x,weight1,bias1,weight2,bias2,weight3,bias3,weight4,bias4,weight5,bias5)
'''
calculating cost and backpropagation
'''
cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))+(regu1*(tf.nn.l2_loss(weight2)))+(regu1*(tf.nn.l2_loss(weight1)))+(regu1*(tf.nn.l2_loss(weight3))+(regu1*(tf.nn.l2_loss(weight4)))+(regu1*(tf.nn.l2_loss(weight5))))
training = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
'''
the softmax function is calculated again because the softmax_cross_entropy_with_logits calculates softmax internally 
and then uses the softmax but it does not change the output
'''
output = tf.nn.softmax(output)
predict = tf.argmax(output, axis=1)
y_test = mnist.test.labels
x_test = mnist.test.images
'''
start session
'''
sess = tf.Session()
sess.run(tf.global_variables_initializer())
t0 = time.time()
for i in range(5000):
    batch_xtrain,batch_ytrain = mnist.train.next_batch(11)
    cost = sess.run(training,feed_dict={x:batch_xtrain,y:batch_ytrain})
    train_accuracy = np.mean(np.argmax(batch_ytrain, axis=1) == sess.run(predict, feed_dict={x: batch_xtrain, y: batch_ytrain}))
    '''
    once dropout is introduced dont take the test images and find the accuracy
    '''
    if(i%100==0):
        print(' iteration number : '+ str(i) + ' training accuracy:  ' + str(train_accuracy))# + '  test accuracy:  '+ str(test_accuracy))
    
print(sess.run(training,feed_dict={x: batch_xtrain,y: batch_ytrain}))
test_accuracy = np.mean(np.argmax(y_test, axis=1) == sess.run(predict, feed_dict={x: x_test, y: y_test}))
print('test accuracy'+str(test_accuracy))
t1 = time.time()
print(t1-t0)
sess.close()