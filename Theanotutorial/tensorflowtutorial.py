# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 11:40:21 2022

@author: Fran
"""

import numpy as np
import tensorflow as tf

#version tf1.0
"""
A = tf.placeholder(tf.float32,shape=(5,5),name='A')
v = tf.placeholder(tf.float32)
w = tf.matmul(A, v)

with tf.Session() as sessions:
    output = sessions.run(w,feed_dict={A:np.random.randn(5,5),v:np.random.randn(5,1)})
    print(output,type(output))

shape = (2,2)
x = tf.Variable(tf.random_normal(shape))
t = tf.Variable(0)
init = tf.initialize_all_variables()
with tf.Session() as sessions:
    out = sessions.run(init)
    print(x.eval())
    print(t.eval())

u = tf.Variable(20.0)
cost = u*u+u+1
train_op = tf.train.GradientDescentOptimizer(0.3).minimize(cost)
init = tf.initialize_all_variables()
with tf.Session() as sessions:
    out = sessions.run(init)
    for i in range(12):
        sessions.run(train_op)
        print( f"i = {i}, cost {cost:.3f} u = {u:.3f}")
"""

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from util import get_normalized_data, y2indicator


def error_rate(p, t):
    return np.mean(p != t)


# copy this first part from theano2.py
def main():
    # step 1: get the data and define all the usual variables
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

    max_iter = 15
    print_period = 50

    lr = 0.00004
    reg = 0.01

    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N // batch_sz

    # add an extra layer just for fun
    M1 = 300
    M2 = 100
    K = 10
    W1_init = np.random.randn(D, M1) / np.sqrt(D)
    b1_init = np.zeros(M1)
    W2_init = np.random.randn(M1, M2) / np.sqrt(M1)
    b2_init = np.zeros(M2)
    W3_init = np.random.randn(M2, K) / np.sqrt(M2)
    b3_init = np.zeros(K)


    # define variables and expressions
    X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))

    # define the model
    Z1 = tf.nn.relu( tf.matmul(X, W1) + b1 )
    Z2 = tf.nn.relu( tf.matmul(Z1, W2) + b2 )
    Yish = tf.matmul(Z2, W3) + b3 # remember, the cost function does the softmaxing! weird, right?

    # softmax_cross_entropy_with_logits take in the "logits"
    # if you wanted to know the actual output of the neural net,
    # you could pass "Yish" into tf.nn.softmax(logits)
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Yish, labels=T))

    # we choose the optimizer but don't implement the algorithm ourselves
    # let's go with RMSprop, since we just learned about it.
    # it includes momentum!
    train_op = tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=0.9).minimize(cost)

    # we'll use this to calculate the error rate
    predict_op = tf.argmax(Yish, 1)

    costs = []
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
                Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]

                session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})
                if j % print_period == 0:
                    test_cost = session.run(cost, feed_dict={X: Xtest, T: Ytest_ind})
                    prediction = session.run(predict_op, feed_dict={X: Xtest})
                    err = error_rate(prediction, Ytest)
                    print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
                    costs.append(test_cost)

    plt.plot(costs)
    plt.show()
    # increase max_iter and notice how the test cost starts to increase.
    # are we overfitting by adding that extra layer?
    # how would you add regularization to this model?


if __name__ == '__main__':
    main()