import numpy as np
import random as shuffle

def svm_loss_naive(W,X,y,reg):#W-> D * C   X -> N * D   y- > 1*C 
	num_classes = W.shape[1]
	num_train = X.shape[0]
	Dw = np.zeros(W.shape)
	loss = 0.0
	for i in range(num_train):
		scores = X[i].dot(W)#有C个类
		label_score = scores[y[i]]
		for j in range(0,num_classes):
			if j == y[i]:
				continue
			margin = scores[j] - label_score + 1
			if margin > 0:
				loss += margin
				Dw[:,j] += X[i] # W 一列为 X一行 （D维）
				Dw[:,y[i]] -= X[i]
	loss /= num_train
	Dw /= num_train
	loss += 0.5 * reg *np.sum(W*W)
	Dw += reg * W

	return loss,Dw

#向量化求法
def svm_loss_vectorized(W,X,y,reg):
	loss = 0.0
	Dw = np.zeros(W.shape)
	num_train = X.shape[0]

	scores = X.dot(W)
	#花式索引 A[[1,2,3,4],[5,6,7,8]] = [A[1,5],A[2,6],A[3,7],A[4,8]]
	margin = scores - scores[np.arange(num_train),y].reshape(num_train,1) + 1
	margin[np.arange(num_train),y] = 0.0
	#将负数去掉
	margin = (margin > 0) * margin
	loss += margin.sum() / num_train
	loss += 0.5 * reg * np.sum(W*W)

	#求Dw
	margin = (margin > 0)*1
	row_sum = np.sum(margin,axis = 1)#将margin求和为一列
	margin[np.arange(num_train),y] = -row_sum
	Dw = X.T.dot(margin) / num_train + reg * W

	return loss,Dw
