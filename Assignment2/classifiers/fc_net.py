import numpy as np

from classifiers.layers import *
from classifiers.layer_utils import *

class TwoLayerNet(object):
    def __init__(self,input_dim=3*32*32,hidden_dim=100,num_classes=10,weight_scale=1e-3,reg=0.0):
        self.param = {}
        self.reg = reg
        self.param['W1'] = np.random.normal(0,weight_scale,size=(input_dim,hidden_dim))#np.random.normal产生一个正态分布
        self.param['b1'] = np.zeros(hidden_dim)
        self.param['W2'] = np.random.normal(0,weight_scale,size=(hidden_dim,num_classes))
        self.param['b2'] = np.zeros(num_classes)

    def loss(self,X,y=None):
        scores = None
        w1 = self.param['W1']
        b1 = self.param['b1']
        w2 = self.param['W2']
        b2 = self.param['b2']
        hidden , affine_relu_cache = affine_relu_forward(X,w1,b1)
        scores , fc_cache = affine_relu_forward(hidden,w2,b2)
        if y is None:
            return scores
        loss , grads = 0,{}
        loss , dout = softmax_loss(scores,y)
        #加上正则
        # loss += 0.5 * self.reg * np.sum(w1 * w1 + w2 * w2) 不能这么写 矩阵大小不匹配
        loss += 0.5 * self.reg * (np.sum(w1 * w1) + np.sum(w2 * w2))
        dhidden, dw2, db2 = affine_relu_backward(dout,fc_cache)
        dx , dw1 , db1 = affine_relu_backward(dhidden,affine_relu_cache)
        #梯度也加上正则
        grads['W1'] = dw1 + self.reg * w1
        grads['W2'] = dw2 + self.reg * w2
        grads['b1'] = db1
        grads['b2'] = db2
        return loss, grads

class FullyConnectedNet(object):
    #{affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        #hidden_dims: A list of integers giving the size of each hidden layer.
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.param = {}

        for i in range(len(hidden_dims)):
            layer = hidden_dims[i]
            self.param['W'+str(i+1)] = np.random.normal(0,weight_scale,size=(input_dim,layer))#D * H1
            self.param['b'+str(i+1)] = np.zeros(layer)
            if self.use_batchnorm:
                self.param['gamma'+str(i+1)] = np.ones(layer)
                self.param['beta'+str(i+1)] = np.zeros(layer)
            input_dim = layer
        #最后一层
        self.param['W'+str(self.num_layers)] = np.random.normal(0, weight_scale, size=(layer, num_classes))
        self.param['b'+str(self.num_layers)] = np.zeros(num_classes)

        #dropout
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode':'train','p':dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_param[0] to the forward pass
        # of the first batch normalization layer, self.bn_param[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_param = []
        if self.use_batchnorm:
            self.bn_param = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.param.items():
            self.param[k] = v.astype(dtype)

    def loss(self,X,y = None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train' #有y则是训练模式，无y则是测试模式
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_param:
                bn_param['mode'] = mode

        scores = None

        hidden = X
        cache = {}
        drop_cache = {}
        for i in range(self.num_layers - 1):
            if self.use_batchnorm:
                hidden, cache[i] = affine_bn_relu_forward(hidden, 
                    self.param['W'+str(i+1)], self.param['b'+str(i+1)], 
                    self.param['gamma'+str(i+1)], self.param['beta'+str(i+1)], 
                    self.bn_param[i])
            else:
                hidden, cache[i] = affine_relu_forward(hidden,self.param['W'+str(i+1)],self.param['b'+str(i+1)])
            if self.use_dropout:
                hidden, drop_cache[i] = dropout_forward(hidden,self.dropout_param)
        out , cache[self.num_layers - 1] = affine_forward(hidden,self.param['W'+str(self.num_layers)],self.param['b'+str(self.num_layers)])
        scores = out
        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        loss , dout = softmax_loss(out,y)
        dhidden , dw, db = affine_backward(dout,cache[self.num_layers-1])
        loss +=  0.5 * self.reg * np.sum(self.param['W'+str(self.num_layers)] * self.param['W'+str(self.num_layers)])
        grads['W'+str(self.num_layers)] = dw + self.reg * self.param['W'+str(self.num_layers)]
        grads['b'+str(self.num_layers)] = db

        #对hidden进行循环backward
        for i in range(self.num_layers - 1,0,-1):
            loss += 0.5 * self.reg * np.sum(self.param['W'+str(i)] * self.param['W'+str(i)])
            if self.use_dropout:
                dhidden = dropout_backward(dhidden, drop_cache[i-1])
            if self.use_batchnorm:
                dhidden, dw, db, dgamma, dbeta = affine_bn_relu_backward(dhidden, cache[i-1])
                grads['gamma'+str(i)] = dgamma
                grads['beta'+str(i)] = dbeta
            else:
                dhidden, dw, db = affine_relu_backward(dhidden, cache[i-1])
            
            grads['W'+str(i)] = dw + self.reg * self.param['W'+str(i)]
            grads['b'+str(i)] = db

        return loss, grads