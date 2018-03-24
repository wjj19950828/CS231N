import pickle as pickle
import numpy as np
import os
from scipy.misc import imread

def load_CIFAR_batch(filename):
	with open(filename,'rb') as f:
		datadict = pickle.load(f,encoding='iso-8859-1')
		X = datadict['data']
		Y = datadict['labels']
		X = X.reshape(10000,3,32,32).transpose(0,2,3,1).astype('float')
		Y = np.array(Y)
		return X,Y

def load_CIFAR10(root):
	xs = []
	ys = []
	for i in range(1,6):
		f = os.path.join(root,'data_batch_%d' % i)
		x , y = load_CIFAR_batch(f)
		xs.append(x)
		ys.append(y)
	#将xs ,ys 连接
	Xtr = np.concatenate(xs)
	Ytr = np.concatenate(ys)
	del x,y
	Xte,Yte = load_CIFAR_batch(os.path.join(root,'test_batch'))
	return Xtr,Ytr,Xte,Yte