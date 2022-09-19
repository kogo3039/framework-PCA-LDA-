import sys
sys.path.append("./")
import numpy as np


class WineLoadtxt:

    def __init__(self, fold, filename1, filename2, delimiter, dtype):
        self.data_in="./data_in/"
        red_data=np.loadtxt(self.data_in+fold+filename1,delimiter=delimiter,dtype=dtype)
        white_data=np.loadtxt(self.data_in+fold+filename2,delimiter=delimiter,dtype=dtype)
        self.red_data = red_data[1:, :]
        self.white_data = white_data[1:, :]
        self.red_X= np.zeros((self.red_data.shape[0],self.red_data.shape[1]))
        self.white_X= np.zeros((self.white_data.shape[0],self.white_data.shape[1]))
    
    def __call__(self, ratio, seed):
       
        for i in range(self.red_X.shape[0]):
            for j in range(self.red_X.shape[1]):
                self.red_X[i, j]=float(self.red_data[i, j])
                
        for i in range(self.white_X.shape[0]):
            for j in range(self.white_X.shape[1]):
                self.white_X[i, j]=float(self.white_data[i, j])
        
        
        train0, test0, train1, test1 \
                                   = self.data_seperate(ratio, seed)

        return train0, test0, train1, test1 
        
        

    def data_seperate(self, ratio, seed):

        X = np.concatenate((self.red_X, self.white_X), axis=0)
        maxim = np.max(X, axis=0)
        minim = np.min(X, axis=0)
        X = (X - minim) / (maxim - minim)
        #X = np.log(1/(X + 1e-7) + 100)
        
        red = X[:self.red_X.shape[0]]
        white = X[self.red_X.shape[0]:]
        
        #print(x0.shape)
        #print(x1.shape)
        
        np.random.seed(seed)
        np.random.shuffle(red)
        np.random.shuffle(white)
        
        
        index0 = int(red.shape[0]*ratio)
        index1 = int(white.shape[0]*ratio)
        
        train0=red[:index0,:]
        test0=red[index0:,:]

        train1=white[:index1,:]
        test1=white[index1:,:]
        
        return train0, test0, train1, test1
        



if __name__=="__main__":
    FOLD="wineDataset/"
    FILENAME1="winequality-red.csv"
    FILENAME2="winequality-white.csv"
    delimiter=";"
    dtype=str
    database=IrisLoadtxt(FOLD,FILENAME1, FILENAME2, delimiter,dtype)
    train0, test0, train1, test1 = database(0.8, 1000)
    
