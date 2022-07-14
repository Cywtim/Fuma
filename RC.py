# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 20:35:34 2021

@author: CHENG
"""


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import Fuma_functions as Ff

learning_rate = 1e-4

err = 1e-6

n = 100

epochs = 50000

data = Ff.read_excel("rc.xls")

D = data[:,1]

L = data[:,6]

R_eff = data[:,8]

R_d = data[:,10]

v = np.log(data[:,14])

n_input = 2

weights = {
    'w_1':tf.Variable(tf.truncated_normal([n_input,128],stddev=0.1)),
    'w_2':tf.Variable(tf.truncated_normal([128,1],stddev=0.1)),
    'w_3':tf.Variable(tf.truncated_normal([2,128],stddev=0.1)),
    'w_4':tf.Variable(tf.truncated_normal([128,1],stddev=0.1)), 
    'w_5':tf.Variable(tf.truncated_normal([3,128],stddev=0.1)),
    'w_6':tf.Variable(tf.truncated_normal([128,1],stddev=0.1))
    }

biases = {
    'b_1':tf.Variable(tf.ones([128])),
    'b_2':tf.Variable(tf.ones([1])),
    'b_3':tf.Variable(tf.ones([128])),
    'b_4':tf.Variable(tf.ones([1])),
    'b_5':tf.Variable(tf.ones([128])),
    'b_6':tf.Variable(tf.ones([1]))
    }


L_R_D_layer = tf.placeholder(tf.float32,[None,n_input])
v_layer = tf.placeholder(tf.float32,[None,1])

layer_1 = tf.nn.relu(tf.matmul(L_R_D_layer,weights['w_1'])+biases['b_1'])

pre = tf.nn.relu(tf.matmul(layer_1,weights['w_2'])+biases['b_2'])

loss = tf.reduce_mean(tf.abs(v_layer - pre)/(v_layer))

train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

accuracy = tf.reduce_mean(tf.abs(v_layer - pre)/(v_layer))


LR_losslist = []
LR_acclist = []
LR_test_acc_list = []



train_L_R = np.concatenate((L[:n].reshape([-1,1]),
                            R_eff[:n].reshape([-1,1])),
                           axis = 1)
train_L_R_D = np.append(train_L_R,D[:n].reshape([-1,1]),axis=1)

train_L = L[:n].reshape([-1,1])
'''
train_m_v = np.concatenate((m[:n].reshape([-1,1]),
                            v[:n].reshape([-1,1])),
                           axis = 1)
train_m_a_v = np.concatenate((m[:n].reshape([-1,1]),
                            a[:n].reshape([-1,1]),
                            v[:n].reshape([-1,1])),
                           axis = 1)
'''
train_v = v[:n].reshape([-1,1])

#############################################
test_L_R = np.concatenate((L[n:].reshape([-1,1]),
                            R_eff[n:].reshape([-1,1])),
                           axis = 1)
test_L_R_D = np.append(test_L_R,D[n:].reshape([-1,1]),axis=1)
test_L = L[n:].reshape([-1,1])
'''
test_m_v = np.concatenate((m[n:].reshape([-1,1]),
                            v[n:].reshape([-1,1])),
                           axis = 1)
test_m_a_v = np.concatenate((m[n:].reshape([-1,1]),
                            a[n:].reshape([-1,1]),
                            v[n:].reshape([-1,1])),
                           axis = 1)
'''
test_v = v[n:].reshape([-1,1])
##############################################################################

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        
        #batch_m_a = train_m_a[epoch*150:(epoch+1)*150-1]
        #batch_F = train_F[epoch*150:(epoch+1)*150-1]
        
        #test_batch_m_a = test_m_a[epoch*150:(epoch+1)*150-1]
        #test_batch_F = test_F[epoch*150:(epoch+1)*150-1]
        
        _,lossval,accval = sess.run([train,loss,accuracy],
                                    feed_dict={L_R_D_layer:train_L_R,
                                               v_layer:train_v})
        
        LR_losslist.append(lossval)
        LR_acclist.append(accval)
        
        test_acc_val = sess.run(accuracy,feed_dict={L_R_D_layer:test_L_R,
                                                    v_layer:test_v})
        LR_test_acc_list.append(test_acc_val)
        
    
    plt.figure(0);plt.plot(LR_losslist[:])
    plt.figure(1);plt.plot(LR_acclist[-100:])
    plt.figure(2);plt.plot(LR_test_acc_list[-100:])


###########################################################################
'''
m_v_layer = tf.placeholder(tf.float32,[None,2])
F_layer2 = tf.placeholder(tf.float32,[None,1])

layer_2 = tf.nn.relu(tf.matmul(m_v_layer,weights['w_3'])+biases['b_3'])

pre2 = tf.nn.relu(tf.matmul(layer_2,weights['w_4'])+biases['b_4'])

loss2 = tf.reduce_mean(tf.abs(F_layer2-pre2))

train2 = tf.train.AdamOptimizer(learning_rate).minimize(loss2)

accuracy2 = tf.reduce_mean(tf.abs(F_layer2-pre2))

mv_losslist = []
mv_acclist = []
mv_test_acc_list = []

with tf.Session() as sess1:
    
    sess1.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        
        #batch_m_v = train_m_v[epoch*150:(epoch+1)*150-1]
        #batch_F = train_F[epoch*150:(epoch+1)*150-1]
        
        #test_batch_m_v = test_m_v[epoch*150:(epoch+1)*150-1]
        #test_batch_F = test_F[epoch*150:(epoch+1)*150-1]
        
        _,lossval,accval = sess1.run([train2,loss2,accuracy2],
                                    feed_dict={m_v_layer:train_m_v,
                                               F_layer2:train_F})
        
        mv_losslist.append(lossval)
        mv_acclist.append(accval)
        
        mv_test_acc_val = sess1.run(accuracy2,feed_dict={m_v_layer:test_m_v,
                                                    F_layer2:test_F})
        mv_test_acc_list.append(mv_test_acc_val)
        
    
    plt.figure(3);plt.plot(mv_losslist[:])
    plt.figure(4);plt.plot(mv_acclist[-100:])
    plt.figure(5);plt.plot(mv_test_acc_list[-100:])
'''
##############################################################################
'''
m_a_v_layer = tf.placeholder(tf.float32,[None,3])
F_layer2 = tf.placeholder(tf.float32,[None,1])

layer_2 = tf.nn.relu(tf.matmul(m_a_v_layer,weights['w_5'])+biases['b_5'])

pre2 = tf.nn.relu(tf.matmul(layer_2,weights['w_6'])+biases['b_6'])

loss2 = tf.reduce_mean(tf.abs(F_layer2-pre2))

train2 = tf.train.AdamOptimizer(learning_rate).minimize(loss2)

accuracy2 = tf.reduce_mean(tf.abs(F_layer2-pre2))

mv_losslist = []
mv_acclist = []
mv_test_acc_list = []

with tf.Session() as sess1:
    
    sess1.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        
        #batch_m_v = train_m_v[epoch*150:(epoch+1)*150-1]
        #batch_F = train_F[epoch*150:(epoch+1)*150-1]
        
        #test_batch_m_v = test_m_v[epoch*150:(epoch+1)*150-1]
        #test_batch_F = test_F[epoch*150:(epoch+1)*150-1]
        
        _,lossval,accval = sess1.run([train2,loss2,accuracy2],
                                    feed_dict={m_a_v_layer:train_m_a_v,
                                               F_layer2:train_F})
        
        mv_losslist.append(lossval)
        mv_acclist.append(accval)
        
        mv_test_acc_val = sess1.run(accuracy2,feed_dict={m_a_v_layer:test_m_a_v,
                                                    F_layer2:test_F})
        mv_test_acc_list.append(mv_test_acc_val)
        
    
    plt.figure(6);plt.plot(mv_losslist[:])
    plt.figure(7);plt.plot(mv_acclist[-100:])
    plt.figure(8);plt.plot(mv_test_acc_list[-100:])
'''







'''
print(" acc_ma: ",np.mean(ma_acclist[-100:])," var ",np.var(ma_acclist[-100:]),
      "\n test_acc_ma: ",np.mean(ma_test_acc_list[-100:])," var ",np.var(ma_test_acc_list[-100:]),
      "\n acc_mv: ",np.mean(mv_acclist[-100:])," var ",np.var(mv_acclist[-100:]),
      "\n test_acc_mv",np.mean(mv_test_acc_list[-100:])," var ",np.var(mv_test_acc_list[-100:])
      )
'''





