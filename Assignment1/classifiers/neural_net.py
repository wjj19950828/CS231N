import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
	#input - fully connected layer - ReLU - fully connected layer - softmax
	def __init__(self,input_size,hidden_size,output_size,std = 1e-4):
		self.params = {}
		self.params['W1'] = std * np.random.randn(input_size,hidden_size)# D * H
		self.params['b1'] = np.zeros(hidden_size)
		self.params['W2'] = std * np.random.randn(hidden_size,output_size)# H * C
		self.params['b2'] = np.zeros(output_size)

	def loss(self,X,y=None,reg=0.0): #X-> N * D
		W1 , b1 = self.params['W1'] , self.params['b1']
		W2 , b2 = self.params['W2'] , self.params['b2']
		N , D = X.shape

		scores = None
		out1 = np.maximum(0,X.dot(W1) + b1) #N * H(relu函数)
		scores = out1.dot(W2) + b2 #N * C
		#如果label没给 直接return
		if y is None:
			return scores
		#compute loss
		loss = None
		correct_class_score = scores[np.arange(N),y].reshape(N,1)
		exp_sum = np.sum(np.exp(scores),axis = 1).reshape(N,1)
		loss = np.sum(np.log(exp_sum) - correct_class_score)
		loss /= N
		loss += 0.5 * reg *(np.sum(W1 * W1) + np.sum(W2 * W2))
		#compute grad
		grads = {}
		margin = np.exp(scores) / exp_sum 
		margin[np.arange(N),y] -= 1 #margin 是softmax传过来的导数
		margin /= N # N * C
		dW2 = out1.T.dot(margin) #W2中 out1为输入
		dW2 += reg * W2
		grads['W2'] = dW2
		grads['b2'] = np.sum(margin,axis = 0)#对b求导 后缀x变为1 因为y=wx + b

		# BP算法反向到达RELU层的值等于到达上一层的反向值乘以上一层的W，
		# 然后乘以本层激励函数的导数即可
		margin1 = margin.dot(W2.T) # N * H ???
		margin1[out1 <= 0] = 0
		dW1 = X.T.dot(margin1) #D * H
		dW1 += reg * W1
		grads['W1'] = dW1
		grads['b1'] = np.sum(margin1,axis = 0)

		return loss,grads

	def train(self,X,y,X_val,y_val,learning_rate=1e-3,learning_rate_decay=0.95,reg=1e-5,num_iters=100,batch_size=200,verbose=False):
		num_train = X.shape[0]
		iterations_per_epoch = max(num_train / batch_size,1)

		loss_history = []
		train_acc_history = []
		val_acc_history = []

		for i in range(num_iters):
			X_batch = None
			Y_batch = None

			mask = np.random.choice(num_train,batch_size,replace = True)
			X_batch = X[mask]
			Y_batch = y[mask]
			loss,grads = self.loss(X_batch,Y_batch,reg = reg)
			loss_history.append(loss)

			#upgrate params
			self.params['W1'] -= learning_rate * grads['W1'] 
			self.params['b1'] -= learning_rate * grads['b1']
			self.params['W2'] -= learning_rate * grads['W2']
			self.params['b2'] -= learning_rate * grads['b2']

			if verbose and i % 100 == 0:
				print('iteration %d / %d : loss %f'%(i,num_iters,loss))
			# Every epoch, check train and val accuracy and decay learning rate.
			if i % iterations_per_epoch == 0:
				train_acc = (self.predict(X_batch) == Y_batch).mean()
				val_acc = (self.predict(X_val) == y_val).mean()
				train_acc_history.append(train_acc)
				val_acc_history.append(val_acc)

				learning_rate *= learning_rate_decay

		return {
			'loss_history': loss_history,
      		'train_acc_history': train_acc_history,
      		'val_acc_history': val_acc_history,
		} 

	def predict(self,X):
		y_pred = None
		out1 = np.maximum(0,X.dot(self.params['W1']) + self.params['b1']) #N * H
		y_pred = np.argmax(out1.dot(self.params['W2']) + self.params['b2'],axis = 1) #从N*C中找出max 变为N*1
		
		return y_pred		