import numpy as np
from random import shuffle

def softmax_loss_naive(W,X,y,reg): #X->N*D  W->D*C
	loss = 0.0
	dW = np.zeros(W.shape)
	num_classes = W.shape[1]
	num_train = X.shape[0]
	for i in range(num_train):
		scores = X[i].dot(W)
		correct_class_score = scores[y[i]]
		exp_sum = np.sum(np.exp(scores))
		loss += np.log(exp_sum) - correct_class_score

		dW[:,y[i]] += -X[i]
		for j in range(num_classes):
			dW[:,j] += (np.exp(scores[j]) / exp_sum) * X[i]

	loss /= num_train
	dW /= num_train

	loss += 0.5 * reg * np.sum(W * W)
	dW += reg * W

	return loss,dW

def softmax_loss_vectorized(W,X,y,reg):
	loss = 0.0
	dW = np.zeros(W.shape)
	num_train = X.shape[0]
	num_classes = W.shape[1]
	scores = X.dot(W)
	correct_class_score = scores[np.arange(num_train),y].reshape(num_train,1)
	exp_sum = np.sum(np.exp(scores),axis = 1).reshape(num_train,1) #求和后为一列(必须要加reshape)
	loss += np.sum(np.log(exp_sum) - correct_class_score)

	margin = np.exp(scores) / exp_sum #构造一个N * C 的margin矩阵 与X.T(D*N)相乘
	margin[np.arange(num_train),y] += -1
	dW = X.T.dot(margin) 

	loss /= num_train
	dW /= num_train

	loss += 0.5 * reg * np.sum(W * W)
	dW += reg * W

	return loss,dW