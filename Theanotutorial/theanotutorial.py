# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:21:22 2022

@author: Fran

Theano Tutorial
"""

# import theano.tensor as T

# c = T.scalar('c')
# v = T.vector('V')
# A = T.matrix('A')

# w = A.dot(v)

# import theano

# matrix_times_vector = theano.function(inputs=[A,v],outputs=w)

# import numpy as np

# A_val = np.array([[1,2],[3,4]])
# v_val = np.array([5,6])

# w_val = matrix_times_vector(A_val,v_val) 

# print(w_val)

# x = theano.shared(20.0, 'x')

# cost = x*x + x +1

# x_update = x - 0.3*T.grad(cost,x)

# train = theano.function(inputs=[], outputs=cost, updates=[(x,x_update)])

# for _ in range(25):
#     cost_val = train()
#     print(cost_val)
    
# print(x.get_value())

import theano
import theano.tensor as T
import numpy as np

import matplotlib.pyplot as plt
from util import get_normalized_data, y2indicator

def error_rate(p,t):
    return np.mean(p != t)

def relu(a):
    return a * (a > 0)

def main():
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    max_iter = 20
    print_period = 10
    lr = 0.0004
    reg = 0.01
    
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)
    
    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N // batch_sz
    
    M = 300
    K = 10
    W1_init = np.random.randn(D,M)/np.sqrt(D)
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M,K)/np.sqrt(M)
    b2_init = np.zeros(K)
    thX = T.matrix('X')
    thT = T.matrix('T')
    W1 = theano.shared(W1_init, 'W1')
    b1 = theano.shared(b1_init, 'b1')
    W2 = theano.shared(W2_init, 'W2')
    b2 = theano.shared(b2_init, 'b2')
    
    thZ = relu( thX.dot(W1) + b1)
    thY = T.nnet.softmax( thZ.dot(W2) + b2)
    
    cost = -(thT * T.log(thY)).sum() + reg*((W1*W1).sum()+(b1*b1).sum()+(W2*W2).sum()+(b2*b2).sum())
    
    prediction = T.argmax(thY,axis=1)
    
    update_W1 = W1 - lr*T.grad(cost,W1)
    update_b1 = b1 - lr*T.grad(cost,b1)
    update_W2 = W2 - lr*T.grad(cost,W2)
    update_b2 = b2 - lr*T.grad(cost,b2)
    
    train = theano.function(
        inputs=[thX,thT],
        updates=[(W1,update_W1),(b1,update_b1),(W2,update_W2),(b2,update_b2)])

    get_prediction = theano.function(
        inputs=[thX,thT],
        outputs = [cost,prediction])
        
    LL = []
    
    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz+batch_sz),]
            Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz+batch_sz),]
            train(Xbatch,Ybatch)
            if j % print_period == 0:
                cost_val, prediction_val = get_prediction(Xtest, Ytest_ind)
                err = error_rate(prediction_val, Ytest)
                print(f"Cost / error at iteration i={i}, j={j}: {cost_val:.3f} / {err:.3f}")
                LL.append(cost_val)
    
    plt.plot(LL)
    plt.show()
    
if __name__ == '__main__':
    main()