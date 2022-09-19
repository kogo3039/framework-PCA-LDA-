import numpy as np
import sys
sys.path.append("./")
from module.classifier import Classifier, LDA, PCA
from module.loadtxt import IrisLoadtxt

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
    FOLD="irisDataset/"
    FILENAME="iris.csv"
    delimiter=","
    dtype=str
    database=IrisLoadtxt(FOLD,FILENAME,delimiter,dtype)
    train0, test0, train1, test1, train2, test2 = database(0.7, seed=7777777)
    
    """
    pca = PCA(train0, test0, train1, test1, train2, test2)
    train0, test0, train1, test1, train2, test2 = pca(interval=3)
    
    """
    lda = LDA(train0, test0, train1, test1, train2, test2)
    train0, test0, train1, test1, train2, test2 = lda(interval=2)
    
    classifier = Classifier(train0, train1, train2)
    acc=accuracy(classifier,test0, test1, test2)
    print(acc)


