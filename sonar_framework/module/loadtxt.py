import sys
sys.path.append("../")
import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class Sonarloadtxt:

    def __init__(self, fold, filename, delimiter, dtype):
        self.data_in="./data_in/"
        self.data=np.loadtxt(self.data_in+fold+filename,delimiter=delimiter,dtype=dtype)
        self.X= np.zeros((self.data.shape[0], self.data.shape[1]))

    def __call__(self, ratio, seed=1234):

        train0, test0, train1, test1 = self.data_seperate(ratio, seed)
        return train0, test0, train1, test1

    def data_seperate(self, ratio, seed):

        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                if self.data[i,j] == 'R':
                    self.X[i, j] = 0
                elif self.data[i,j] == 'M':
                    self.X[i,j] = 1
                else:
                    #self.X[i,j] = np.log(float(self.data[i,j]) + 10)
                    # 50: 82  // LDA: 70:69
                    #self.X[i,j] = float(self.data[i,j])
                    # 50: 82 // LDA 65: 82
                    self.X[i,j] = np.log(1/(float(self.data[i,j])+1e-6) + 50)
                    # 55: 69 //LDA 65: 95  
        #self.X = sigmoid(self.X)
        #print(self.X)
        maxim = np.max(self.X, axis=0)
        minim = np.min(self.X, axis=0)
        mean = np.mean(self.X, axis=0)
        self.X = (self.X - minim) / (maxim - minim)
        self.X = np.log(1/(self.X + 1e-7) + 100)
        #self.X = np.exp(self.X + 1)
        #print(self.X)
        
                    

        rData = self.X[:97, :-1]
        mData = self.X[97:, :-1]
        np.random.seed(seed) 
        np.random.shuffle(rData)
        np.random.seed(seed) 
        np.random.shuffle(mData)

        index0 = int(rData.shape[0]*ratio)
        index1 = int(mData.shape[0]*ratio)

        train0=rData[:index0,:]
        test0=rData[index0:,:]

        train1=mData[:index1,:]
        test1=mData[index1:,:]

        return train0, test0, train1, test1




if __name__=="__main__":
    FOLD="sonarDataset/"
    FILENAME="sonar.all-data.csv"
    delimiter=","
    dtype=str
    database=Sonarloadtxt(FOLD,FILENAME,delimiter,dtype)
    train0,test0,train1,test1=database(0.8)
    
