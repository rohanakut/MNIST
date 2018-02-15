# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 06:37:16 2017

@author: Rohan-PC
"""

"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import time
import numpy as np
#import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32,[None,785])
y = tf.placeholder(tf.float32,[None,10])
#regu = tf.placeholder(tf.float32)
regu=0.0013
a=[4,3,2]
#for i in range(3):
#    regu.append(pow(10,a[i]))
accuracy_val=[]
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels
length = len(x_train)
one = np.ones((length,1))
x_train = np.append(x_train,one,axis=1)
one = np.ones((len(x_test),1))
x_test = np.append(x_test,one,axis=1)
theta  = tf.Variable(tf.zeros([785, 10]))
x_train = x_train[0:10000]
y_train = y_train[0:10000]
size = x_train.shape[0]
theta1 = theta[1:785,:]
h_theta = tf.nn.softmax(tf.matmul(x,theta))
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h_theta), reduction_indices=[1]))+(regu*(tf.nn.l2_loss(theta1)))
LEARNING_RATE = 0.1
TRAIN_STEPS = 2500
#theta1 = theta[1:785,:]
training  = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)
#print(theta)
correct_prediction = tf.equal(tf.argmax(h_theta,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
t0 = time.time()
for i in range(2500):
    batch_xtrain,batch_ytrain = mnist.train.next_batch(22)
    one = np.ones((len(batch_xtrain),1))
    batch_xtrain = np.append(batch_xtrain,one,axis=1)
    sess.run(training,feed_dict={x:batch_xtrain,y:batch_ytrain})#,regu:1e-2})   
    if i%100 == 0:
        print('Training Step:' + str(i) + '  Loss = ' + str(sess.run(cost, {x: batch_xtrain, y: batch_ytrain})))#,regu : 1e-2})))
        print(sess.run(accuracy,feed_dict = {x : x_test,y: y_test}))#,regu : 1e-2}))
        accuracy_val.append(sess.run(accuracy,feed_dict = {x : x_test,y: y_test}))#,regu : 1e-3}))
 #       print(sess.run(cost,feed_dict = {x : x_test,y: y_test}))#,regu : 1e-2}))
t1 = time.time()
print(t1-t0,'secs')
sess.close()
