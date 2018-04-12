#!/bin/bash/env python
#-*- coding: utf-8
__author__ = 'ZhangJin'

import numpy as np
from sklearn.datasets import make_moons

class LR:
    def __init__(self):
        self.dim = 2
 #       self.w = np.random.random(self.dim)
        self.w = np.array([1.0,1.0])
        self.b = 0
        self.eta = 0.2


    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))

    def logistic_regression(self,x,y,eta):
        itr = 0
        self.eta = eta
        row, column = np.shape(x)
        xpts = np.linspace(-1.5, 2.5)
        while itr <= 500:
            fx = np.dot(self.w, x.T) + self.b
            hx = self.sigmoid(fx)
            t = (hx-y)
            s = [[i[0]*i[1][0],i[0]*i[1][1]] for i in zip(t,x)]
            gradient_w = np.sum(s, 0)/row * self.eta #np.sum的三种，sum(x,0)所有x的列元素相加，sum(x,1)行元素相加
            gradient_b = np.sum(t, 0)/row * self.eta
            self.w -= gradient_w
            self.b -= gradient_b
            ypts = (self.w[0] * xpts + self.b) / (-self.w[1])
            if itr%100 == 0:
                plt.figure()
                for i in range(250):
                    plt.plot(x[i, 0], x[i, 1], col[y[i]] + 'o')
                plt.ylim([-1.5,1.5])
                plt.plot(xpts,ypts, 'g*', lw = 2)
                plt.title('eta = %s, Iteration = %s\n' % (str(eta), str(itr)))
                plt.savefig('p_N%s_it%s' % (str(row), str(itr)), dpi=200, bbox_inches='tight')
                # plt.plot(5.50113924e-01, -9.35132373e-01, 'b*', lw=3)
            itr += 1
        plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x, y = make_moons(250, noise=0.25)
    col = {0:'r',1:'b'}
    lr = LR()
    print(x)
    print(y)
    lr.logistic_regression(x,y,eta=1.2)