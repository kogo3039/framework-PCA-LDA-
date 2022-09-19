import numpy as np
import sys
sys.path.append("./")
from module.classifier import Classifier, LDA, PCA
from module.loadtxt import WineLoadtxt

def accuracy(classifier, *args):
    tests = list(args)
    cnt=np.zeros(len(tests))

    for ans,test in enumerate(tests):
  
        for t in test:
            y=classifier(t)
            
            if ans==y:
                cnt[ans]+=1

    for i in range(len(tests)):
        cnt[i] = cnt[i]/len(tests[i])*100
        #print(len(tests[i]))

    return cnt






if __name__=="__main__":
    FOLD="wineDataset/"
    FILENAME1="winequality-red.csv"
    FILENAME2="winequality-white.csv"
    delimiter=";"
    dtype=str
    database=WineLoadtxt(FOLD,FILENAME1, FILENAME2, delimiter,dtype)
    train0, test0, train1, test1 = database(0.7, seed=666)
    """
    pca = PCA(train0, test0, train1, test1)
    train0, test0, train1, test1 = pca(interval=9)
    
    """
    lda = LDA(train0, test0, train1, test1)
    train0, test0, train1, test1 = lda(interval=5)
    
    classifier = Classifier(train0, train1)
    acc=accuracy(classifier,test0, test1)
    print(acc)


