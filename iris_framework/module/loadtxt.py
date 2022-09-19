import sys
sys.path.append("./")
import numpy as np


class IrisLoadtxt:

    def __init__(self, fold, filename, delimiter, dtype):
        self.data_in="./data_in/"
        self.data=np.loadtxt(self.data_in+fold+filename,delimiter=delimiter,dtype=dtype)
        self.X= np.zeros((self.data.shape[0],self.data.shape[1]))

    def __call__(self, ratio, seed):

        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                if self.data[i, j] == "Iris-setosa":
                    self.X[i, j] = 0
                elif self.data[i, j] == "Iris-versicolor":
                    self.X[i, j] = 1
                elif self.data[i, j] == "Iris-virginica":
                    self.X[i, j] = 2
                else:
                    self.X[i, j]=float(self.data[i, j]) 
        

        train0, test0, train1, test1, train2, test2 \
                                   = self.data_seperate(ratio, seed)

        return train0, test0, train1, test1, train2, test2 

        

    def data_seperate(self, ratio, seed):
        
        self.X = self.X[:, :-1]
        
        maxim = np.max(self.X, axis=0)
        minim = np.min(self.X, axis=0)
        X = (self.X - minim) / (maxim - minim)
        #X = np.log(1/(X + 1e-7) + 100)
        
        x0 = self.X[:50]
        x1 = self.X[50:100]
        x2 = self.X[100:]
        #print(x0.shape)
        #print(x1.shape)
        
        np.random.seed(seed)
        np.random.shuffle(x0)
        np.random.shuffle(x1)
        np.random.shuffle(x2)
        
        index0 = int(x0.shape[0]*ratio)
        index1 = int(x1.shape[0]*ratio)
        index2 = int(x2.shape[0]*ratio)

        train0=x0[:index0,:]
        test0=x0[index0:,:]

        train1=x1[:index1,:]
        test1=x1[index1:,:]

        train2=x2[:index2,:]
        test2=x2[index2:,:]

        return train0, test0, train1, test1, train2, test2




if __name__=="__main__":
    FOLD="irisDataset/"
    FILENAME="iris.csv"
    delimiter=","
    dtype=str
    database=IrisLoadtxt(FOLD,FILENAME,delimiter,dtype)
    train0, test0, train1, test1, train2, test2=database(0.8)
    
