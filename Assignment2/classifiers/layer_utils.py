#from layers import *
from classifiers.layers import *

def affine_relu_forward(x,w,b):
    out_a , cache_a = affine_forward(x,w,b)
    out , cache_relu = relu_forward(out_a)
    cache = (cache_a,cache_relu)
    return out,cache

def affine_relu_backward(dout,cache):
    cache_a , cache_relu = cache
    dx = relu_backward(dout,cache_relu)
    dx , dw ,db = affine_backward(dx,cache_a)
    return dx,dw,db

def affine_bn_relu_forward(x,w,b,gamma,beta,bn_param):
    a, fc_cache = affine_forward(x, w, b)
    a_bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(a_bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

def affine_bn_relu_backward(dout,cache):
    fc_cache, bn_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dgamma, dbeta = batchnorm_backward(da, bn_cache)
    dx, dw, db = affine_backward(dx, fc_cache)
    return dx, dw, db, dgamma, dbeta
