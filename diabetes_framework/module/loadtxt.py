import sys
sys.path.append("./")
import numpy as np


class Diabetesloadtxt:

    def __init__(self, fold, filename, delimiter, dtype):
        self.data_in="./data_in/"
        self.data=np.loadtxt(self.data_in+fold+filename,delimiter=delimiter,dtype=dtype)
        self.X= np.zeros((self.data.shape[0]-1,self.data.shape[1]))

    def __call__(self, ratio, seed=1234):

        self.data=self.data[1:,:]

        for i in range(self.X.shape[0]):
            for k in range(self.X.shape[1]):
                self.X[i,k]=float(self.data[i,k])
        

        train0,test0,train1,test1= self.data_seperate(ratio, seed)

        return train0,test0,train1,test1

        

    def data_seperate(self, ratio, seed):

        x0=np.array([])
        x1=np.array([])
        cnt=np.zeros(2)
                
        for i in range(self.X.shape[0]):
            if(self.X[i,-1]==0):
                x0=np.append(x0,self.X[i,:-1])
                cnt[0]+=1
            else:
                x1=np.append(x1,self.X[i,:-1])
                cnt[1]+=1

        x0=x0.reshape(int(cnt[0]),self.X.shape[1]-1)
        x1=x1.reshape(int(cnt[1]),self.X.shape[1]-1)
        #print(x0.shape)
        #print(x1.shape)

        X = np.concatenate((x0, x1), axis=0)

        maxim = np.max(X, axis=0)
        minim = np.min(X, axis=0)
        mean = np.mean(X, axis=0)
        X = (X - minim) / (maxim - minim)
        #X = np.log(1/(X + 1e-7) + 100)

        x0 = X[:int(cnt[0])]
        x1 = X[int(cnt[0]):]
        #print(x0.shape)
        #print(x1.shape)
        
        np.random.seed(seed)
        np.random.shuffle(x0)
        np.random.shuffle(x1)

        index0 = int(x0.shape[0]*ratio)
        index1 = int(x1.shape[0]*ratio)

        train0=x0[:index0,:]
        test0=x0[index0:,:]

        train1=x1[:index1,:]
        test1=x1[index1:,:]

        return train0, test0, train1, test1




if __name__=="__main__":
    FOLD="diabetesDataset/"
    FILENAME="diabetes.csv"
    delimiter=","
    dtype=str
    database=Diabetesloadtxt(FOLD,FILENAME,delimiter,dtype)
    train0,test0,train1,test1=database(0.8)
    
