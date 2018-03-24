import numpy as np

class KNN(object):
    def __init__(self):
        pass

    def train(self,X,Y):#对于KNN 只是把数据存储即可
        self.X_train = X
        self.Y_train = Y

    def predict(self,X,k=1,num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loops(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists,k)

    def compute_distances_no_loops(self,X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test,num_train))

        dists += np.sum(np.multiply(X,X),axis = 1,keepdims = True).reshape(num_test,1)
        dists += np.sum(np.multiply(self.X_train,self.X_train),axis = 1,keepdims = True).reshape(1,num_train)
        dists += -2 * np.dot(X,self.X_train.T)
        dists = np.sqrt(dists)
        return dists

    #def compute_distances_one_loops(self,X):

    #def compute_distances_two_loops(self,X):

    def predict_labels(self,dists,k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            closest_y = self.Y_train[np.argsort(dists[i])[:k]]#argsort从小到大返回index
            y_pred[i] = np.argmax(np.bincount(closest_y))
             # 我们可以看到x中最大的数为7，因此bin的数量为8，那么它的索引值为0->7
            # x = np.array([0, 1, 1, 3, 2, 1, 7])
            # # 索引0出现了1次，索引1出现了3次......索引5出现了0次......
            # np.bincount(x)
            # #因此，输出结果为：array([1, 3, 1, 1, 0, 0, 0, 1])
        return y_pred