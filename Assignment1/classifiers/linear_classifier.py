import numpy as np
from classifiers.linear_svm import *
from classifiers.softmax import *

class LinearClassifier(object):

	def __init__(self):
		self.W = None

	def train(self,X,Y,lr = 1e-3,reg = 1e-5,num_iters = 100,batch_size = 200,verbose = False):
		#verbose: (boolean) If true, print progress during optimization.
		num_train,dim = X.shape
		num_classes = np.max(Y) + 1 
		if self.W is None:
			self.W = 0.001 * np.random.randn(dim,num_classes)

		loss_history = []
		for it in range(num_iters):
			X_batch = None
			Y_batch = None
			mask = np.random.choice(num_train,batch_size,replace = True)
			X_batch = X[mask]
			Y_batch = Y[mask]
			loss, grad = self.loss(X_batch,Y_batch,reg)
			loss_history.append(loss)
			self.W -= lr * grad

			if verbose and it % 100 == 0:
				print ('iteration %d / %d: loss %f' % (it, num_iters, loss))

		return loss_history

	def loss(self,X_batch,Y_batch,reg):
		pass

	def predict(self,X):
		Y_pred = np.zeros(X.shape[0])
		Y_pred = np.argmax(X.dot(self.W),axis = 1)

		return Y_pred

class LinearSVM(LinearClassifier):
	def loss(self,X_batch,Y_batch,reg):
		return svm_loss_vectorized(self.W,X_batch,Y_batch,reg)

class Softmax(LinearClassifier):
	def loss(self,X_batch,Y_batch,reg):
		return softmax_loss_vectorized(self.W,X_batch,Y_batch,reg)



