# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 14:33:48 2021

@author: CHENG
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import Fuma_functions as Ff

learning_rate = 1e-4

err = 1e-6

n = 100

epochs = 10000

data = Ff.read_excel("rc.xls")

D = data[:,1]

L = data[:,6] #* 3.828e26

R_eff = data[:,8]

R_d = data[:,10]

R_H = data[:,13]

R = R_H #* 3.085677581491367e19

v = data[:,14] #* 1000

state = np.random.get_state()
np.random.shuffle(R)

np.random.set_state(state)
np.random.shuffle(L)

np.random.set_state(state)
np.random.shuffle(v)


weights = {
    'w_1':tf.Variable(tf.truncated_normal([2,64],stddev=0.1)),
    'w_2':tf.Variable(tf.truncated_normal([64,64],stddev=0.1)),
    'w_3':tf.Variable(tf.truncated_normal([64,64],stddev=0.1)),
    'w_4':tf.Variable(tf.truncated_normal([64,64],stddev=0.1)), 
    'w_5':tf.Variable(tf.truncated_normal([64,64],stddev=0.1)),
    'w_6':tf.Variable(tf.truncated_normal([64,1],stddev=0.1))
    }

biases = {
    'b_1':tf.Variable(tf.ones([64])),
    'b_2':tf.Variable(tf.ones([64])),
    'b_3':tf.Variable(tf.ones([64])),
    'b_4':tf.Variable(tf.ones([64])),
    'b_5':tf.Variable(tf.ones([64])),
    'b_6':tf.Variable(tf.ones([1]))
    }


L_R_D_layer = tf.placeholder(tf.float32,[None,2])
v_layer = tf.placeholder(tf.float32,[None,1])

layer_1 = tf.nn.relu(tf.matmul(L_R_D_layer,weights['w_1'])+biases['b_1'])

layer_2 = tf.nn.relu(tf.matmul(layer_1,weights['w_2'])+biases['b_2'])

#layer_3 = tf.nn.relu(tf.matmul(layer_2,weights['w_3'])+biases['b_3'])

#layer_4 = tf.nn.relu(tf.matmul(layer_3,weights['w_4'])+biases['b_4'])

#layer_5 = tf.nn.relu(tf.matmul(layer_4,weights['w_5'])+biases['b_5'])

pre = tf.nn.relu(tf.matmul(layer_2,weights['w_6'])+biases['b_6'])

loss = tf.reduce_mean(tf.abs(v_layer - pre)/(v_layer))

train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

accuracy = tf.reduce_mean(tf.abs(v_layer - pre)/(v_layer))


LR_losslist = []
LR_acclist = []
LR_test_acc_list = []



train_L_R = np.concatenate((L[:n].reshape([-1,1]),
                            R[:n].reshape([-1,1])),
                           axis = 1)

train_L_v = np.concatenate((L[:n].reshape([-1,1]),
                            v[:n].reshape([-1,1])),
                           axis = 1)
train_R_v = np.concatenate((R[:n].reshape([-1,1]),
                            v[:n].reshape([-1,1])),
                           axis = 1)

train_v = v[:n].reshape([-1,1])

train_L = L[:n].reshape([-1,1])

train_R = R[:n].reshape([-1,1])
#-----------------------------------------------------------------------------
test_L_R = np.concatenate((L[n:].reshape([-1,1]),
                            R[n:].reshape([-1,1])),
                           axis = 1)


test_L_v = np.concatenate((L[n:].reshape([-1,1]),
                            v[n:].reshape([-1,1])),
                           axis = 1)
test_R_v = np.concatenate((R[n:].reshape([-1,1]),
                            v[n:].reshape([-1,1])),
                           axis = 1)

test_v = v[n:].reshape([-1,1])

test_L = L[n:].reshape([-1,1])

test_R = R[n:].reshape([-1,1])
##############################################################################

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        
        #batch_m_a = train_m_a[epoch*150:(epoch+1)*150-1]
        #batch_F = train_F[epoch*150:(epoch+1)*150-1]
        
        #test_batch_m_a = test_m_a[epoch*150:(epoch+1)*150-1]
        #test_batch_F = test_F[epoch*150:(epoch+1)*150-1]
        
        _,lossval,accval = sess.run([train,loss,accuracy],
                                    feed_dict={L_R_D_layer:train_R_v,
                                               v_layer:train_L})
        
        LR_losslist.append(lossval)
        LR_acclist.append(accval)
        
        test_acc_val = sess.run(accuracy,feed_dict={L_R_D_layer:test_R_v,
                                                    v_layer:test_L})
        LR_test_acc_list.append(test_acc_val)
        
    
    plt.figure(0);plt.plot(LR_losslist[:])
    plt.figure(1);plt.plot(LR_acclist[:])
    plt.figure(2);plt.plot(LR_test_acc_list[-100:])

