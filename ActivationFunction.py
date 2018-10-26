# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 17:32:06 2018

@author: Katie
"""  
import numpy as np
from numba import jit


class ActivationFunction(object):

    @staticmethod
    def sigmoid(v):
        return 1 / (1 + np.exp(np.multiply(-1, v)))

    @staticmethod
    def sigmoid_deriv(v):
        return v * (1-v)

    @staticmethod
    def tanh(v):
        exp = np.exp(np.multiply(2, v))
        return (exp-1) / (exp+1)

    @staticmethod
    def tanh_deriv(v):
        return 1 - v*v


    @staticmethod
    @jit
    def softmax(v, shift=False):
        # when given large logits, softmax might blow up
        # to improve numerical stability, shift the logits by the maximum among them to make all logits negative
        # shifting the logits does not affect the softmax derivatives
        if shift:
            v = v - np.max(v)
        return np.exp(v) / np.sum(np.exp(v))

    @staticmethod
    @jit
    def softmax_deriv(v):
        length = np.shape(v)[0]
        m = np.zeros((length, length))
        
        for r in range(0, length):
            for c in range(0, length):
                if r==c:
                    m[r][c] = v[r] * (1 - v[c])
                else:
                    m[r][c] = v[r] * (0 - v[c])
        return m

    @staticmethod
    def ReLU(ary):
        # ary:  it has to be a numpy arary of arbitrary dimension
        return ary * (ary>0)  # True and False are treated as 1 and 0 respectively when used in expressions
    
    @staticmethod
    def ReLU_deriv(ary):
        # ary:  it has to be a numpy arary of arbitrary dimension
        return 1 * (ary>0)  # True and False are treated as 1 and 0 respectively when used in expressions

    @staticmethod
    @jit
    def Leaky_ReLU(ary):
        # ary:  it has to be a numpy arary of arbitrary dimension
        return (ary * (ary>0.0)) + (0.01 * ary * (ary<0.0))

    @staticmethod
    @jit
    def Leaky_ReLU_deriv(ary):
        # ary:  it has to be a numpy arary of arbitrary dimension
        return (1.0 * (ary>0.0)) + (0.01 * (ary<0.0))
    
    
    