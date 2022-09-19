import numpy as np
import sys
sys.path.append("./")
from module.classifier import Classifier, LDA, PCA
from module.loadtxt import Sonarloadtxt

def accuracy(classifier, *args):
    tests = list(args)
    cnt=np.zeros(len(tests))

    for ans, test in enumerate(tests):
  
        for t in test:
            #print(t.shape)
            y = classifier(t)
            if ans == y:
                cnt[ans] += 1

    for i, test in enumerate(tests):
        cnt[i] = cnt[i] / len(test) * 100

    return cnt






if __name__=="__main__":
    FOLD="sonarDataset/"
    FILENAME="sonar.all-data.csv"
    delimiter=","
    dtype=str
    database=Sonarloadtxt(FOLD,FILENAME,delimiter,dtype)
    train0,test0,train1,test1=database(ratio=0.7, seed=7777)
    """
    pca = PCA(train0,test0,train1,test1)
    train0,test0,train1,test1 = pca(interval=30)
    """
    lda = LDA(train0,test0,train1,test1)
    train0,test0,train1,test1 = lda(interval=35)
    
    classifier = Classifier(train0, train1)
    acc=accuracy(classifier,test0,test1)
    print(acc)
    

