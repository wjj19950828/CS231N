#将每一层分成一个模块
import numpy as np

def affine_forward(x,w,b):# X-> N * D, w-> D * M,b-> M
    out = None
    out = x.reshape(x.shape[0],-1).dot(w) + b
    cache = (x,w,b)
    return out,cache

def affine_backward(dout,cache):#dout为上一层的求导结果
    x,w,b = cache
    dx,dw,db = None,None,None
    dx = dout.dot(w.T).reshape(x.shape[0],-1)
    dw = x.reshape(x.shape[0],-1).T.dot(dout)
    db = np.sum(dout,axis = 0)
    return dx,dw,db

def relu_forward(x):
    out = None
    out = np.maximum(0,x)
    cache = x
    return out,cache

def relu_backward(dout,cache):
    x = cache
    dx = None
    dx = (x > 0) * dout

    return dx

def batchnorm_forward(x,gamma,beta,bn_param):
    # - bn_param: Dictionary with the following keys:
 #      - mode: 'train' or 'test'; required
 #      - eps: Constant for numeric stability
 #      - momentum: Constant for running mean / variance.
 #      - running_mean: Array of shape (D,) giving running mean of features
 #      - running_var Array of shape (D,) giving running variance of features

    mode = bn_param['mode']
    eps = bn_param.get('eps',1e-5)#dictionary.get() 后面参数为默认值
    momentum = bn_param.get('momentum',0.9)

    N , D = x.shape
    running_mean = bn_param.get('running_mean',np.zeros(D,dtype = x.dtype))
    running_var = bn_param.get('running_var',np.zeros(D,dtype = x.dtype))

    out,cache = None,None
    if mode == 'train':
        sample_mean = np.mean(x,axis = 0) #1 * D
        sample_var = np.var(x,axis = 0)
        xhat = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * xhat + beta
        cache = (gamma,x,sample_mean,sample_var,eps,xhat)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

    elif mode == 'test':
        xhat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * xhat + beta

    else:
        raise ValueError('Invalid BN mode %s '% mode)

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out,cache

def batchnorm_backward(dout,cache):
    dx , dgamma , dbeta = None,None,None #backward的返回值为forward层的参数的梯度
    gamma,x,sample_mean,sample_var,eps,xhat = cache
    N = x.shape[0]

    a = np.sqrt(sample_var + eps)
    dxhat = dout * gamma
    dgamma = np.sum(dout * xhat,axis = 0)# 1 * D 
    dbeta = np.sum(dout,axis = 0)#1 * D

    #dx分为三部分
    dvar = np.sum((dxhat*(x - sample_mean)*-0.5 / a**3),axis = 0)#1 * D
    dmean = np.sum(dxhat / -a,axis = 0) + np.sum(dvar * -2 * (x - sample_mean),axis = 0) / N
    dx = dxhat / a + dvar * 2 * (x - sample_mean) / N + dmean / N

    return dx,dgamma,dbeta

#inverted dropout
def dropout_forward(x,dropout_param):
    # - dropout_param: A dictionary with the following keys:
 #      - p: Dropout parameter. We drop each neuron output with probability p.
 #      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
 #        if the mode is test, then just return the input.
 #      - seed: Seed for the random number generator. Passing seed makes this
 #        function deterministic, which is needed for gradient checking but not
 #        in real networks.
    p , mode = dropout_param['p'] , dropout_param['mode']
    if 'seed'  in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None
    if mode == 'train':
        #随机生成N * D维的数据 将小于P的数据设为1 因为随机数组小于0.1的数据大概比例为0.1
        mask = (np.random.rand(x.shape[0],x.shape[1]) < p ) / p
        out = x * mask

    elif mode == 'test':
        out = x

    cache = (dropout_param,mask)
    out = out.astype(x.dtype,copy=False)

    return out,cache

def dropout_backward(dout,cache):
    dropout_param , mask = cache
    mode = dropout_param['mode']
    dx = None
    if(mode == 'train'):
        dx = dout * mask
    elif(mode == 'test'):
        dx = dout

    return dx

def conv_forward_naive(x,w,b,conv_param):
    # Input:
 #    - x: Input data of shape (N, C, H, W)
 #    - w: Filter weights of shape (F, C, HH, WW)
 #    - b: Biases, of shape (F,)
 #    - conv_param: A dictionary with the following keys:
 #      - 'stride': The number of pixels between adjacent receptive fields in the
 #        horizontal and vertical directions.
 #      - 'pad': The number of pixels that will be used to zero-pad the input.
     out = None
     stride = conv_param['stride']
     pad = conv_param['pad']
     N,C,H,W = x.shape
     F,C,HH,WW = w.shape

     H_out = int((H + 2*pad - HH) / stride + 1)
     W_out = int((W + 2*pad - WW) / stride + 1)
     out = np.zeros((N,F,H_out,W_out))
     #给x加value=0的padding
     #(0,)表示(0,0)
     x_pad = np.pad(x,((0,),(0,),(pad,),(pad,)),mode = 'constant',constant_value = 0)
     for i in range(H_out):
         for j in range(W_out):
             x_padded_mask = x_pad[:,:,i*pad:i*pad + HH,j*pad:j*pad + WW] #N*C*HH*WW
             for k in range(F):
                 #将每个通道数相加
                 out[:,k,i,j] = np.sum(x_padded_mask*w[k,:,:,:],axis = (1,2,3) )
     out += b[None,:,None,None]#用None填充 添加一个维度 默认值为1

     cache = (x,w,b,conv_param)
     return out,cache

def conv_backward_naive(dout,cache):
    x, w , b , conv_param = cache
    stride , pad = conv_param['stride'] , conv_param['pad']
    N,C,H,W = x.shape
    F,C,HH,WW = w.shape
    H_out = int((H + 2*pad - HH) / stride + 1)
    W_out = int((W + 2*pad - WW) / stride + 1)
    out = np.zeros((N,F,H_out,W_out))
     #给x加value=0的padding
     #(0,)表示(0,0)
    x_pad = np.pad(x,((0,),(0,),(pad,),(pad,)),mode = 'constant',constant_value = 0)
    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.sum(dout,axis = (0,2,3)) #(F,)
    for i in range(H_out):
        for j in range(W_out):
            x_padded_mask = x_pad[:,:,i*stride:i*stride+HH,j*stride:j*stride+WW]
            for k in range(F):
                 #将dout中每一个单独的值取出来，与对应的x_padded_mask做乘积并累加
                dw[k,:,:,:] += np.sum((dout[:,k,i,j])[:,None,None,None] * x_padded_mask,axis = 0)
            for n in range(N):
                 #用dout中的每一个值和原卷积核进行元素乘，然后对dx的对应区域进行叠加
                dx_pad[n,:,i*stride:i*stride+HH,j*stride:j*stride+WW] += np.sum((dout[n,:,i,j])[:,None,None,None] * w,axis = 0)

    dx = dx_pad[:,:,pad:-pad,pad:-pad]
    return dx,dw,db

def maxpool_forward_naive(x,pool_param):
    # - pool_param: dictionary with the following keys:
 #      - 'pool_height': The height of each pooling region
 #      - 'pool_width': The width of each pooling region
 #      - 'stride': The distance between adjacent pooling regions

    out = None
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N,C,H,W = x.shape
    H_out = int((H - pool_height) / stride + 1)
    W_out = int((W - pool_width) / stride + 1)
    out = np.zeros(N,C,H_out,W_out)
    for i in range(H_out):
        for j in range(W_out):
            x_padded_mask = x[:,:,i*stride:i*stride+pool_height,j*stride:j*stride+pool_width]
            out[:,:,i,j] = np.max(x_padded_mask,axis = (2,3))#在后面两个维度上面求max

    cache = (x,pool_param)
    return out , cache

def maxpool_backward_naive(dout,cache):
    x , pool_param = cache
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N,C,H,W = x.shape
    H_out = int((H - pool_height) / stride + 1)
    W_out = int((W - pool_width) / stride + 1)
    dx = np.zeros_like(x)
    #其backword过程与卷积类似
    for i in range(H_out):
        for j in range(W_out):
            x_padded_mask = x[:,:,i*stride:i*stride+pool_height,j*stride:j*stride+pool_width]
            x_pool_max = np.max(x_padded_mask,axis = (2,3))
            #找到x_padded_mask中对应的max 并置为1
            dx_mask = (x_padded_mask == (x_pool_max)[None,None,:,:])
            dx[:,:,i*stride:i*stride+pool_height,j*stride:j*stride+pool_width] += np.sum(dx_mask * (dout[:,:,i,j])[:,:,None,None] ,axis = 0)

    return dx

#此函数将(N,C,H,W)变为(N*H*W,C)
def spatial_batchnorm_forward(x,gamma,beta,bn_param):
    out , cache = None,None
    N , C ,H , W = x.shape
    a , cache = batchnorm_forward(x.transpose(0,2,3,1).reshape((N*H*W,C)),gamma,beta,bn_param)
    out = a.reshape(N,H,W,C).transpose(0,3,1,2)#再将a变为原来的形状
    return out,cache

def spatial_batchnorm_backward(dout,cache):
    dx,dgamma,dbeta = None,None,None
    N,C,H,W = dout.shape
    dx_bn,dgamma,dbeta = batchnorm_backward(dout.transpose(0,2,3,1).reshape((N*H*W,C)),cache)
    dx = dx_bn.reshape(N,H,W,C).transpose(0,3,1,2)
    return dx,dgamma,dbeta

def svm_loss(x,y):#x->(N,C)
    N = x.shape[0]
    correct_class_scores = x[np.arange(N),y].reshape(N, 1)
    margins = np.maximum(0,x - correct_class_scores + 1)
    margins[np.arange(N),y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0,axis = 1)#对x求导 在(N,yi) = -num_pos ,其他大于0位置为1
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N),y] -= num_pos
    dx = dx / N 
    return  loss,dx

def softmax_loss(x,y):
    #防止计算量过大，先将x减去max(x)
    shifted_logits = x - np.max(x,axis = 1,keepdims = True)
    Z = np.sum(np.exp(shifted_logits),axis = 1,keepdims = True)
    loss_sum = np.log(Z) - shifted_logits
    N = x.shape[0] 
    loss = np.sum(loss_sum[np.arange(N),y]) / N
    margins = np.exp(shifted_logits) / Z
    margins[np.arange(N),y] -= 1
    dx = margins / N
    return loss,dx






