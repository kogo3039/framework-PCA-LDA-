import numpy as np

class Classifier:

    def __init__(self, *args):
        self.args = list(args)
        self.total = np.sum([a.shape[1] for a in args])

    def __call__(self, x):

        self.means=[]
        self.covs=[]
        self.invs=[]
        self.dets=[]
        
        if self.args[0].shape[1] == 1:
            for arg in self.args:
                mean = np.mean(arg)
                cov = np.var(arg)
                inv = 1 / cov
                
                self.means.append(mean)
                self.covs.append(cov)
                self.invs.append(inv)
                
            gs=[]
            for i in range(len(self.args)):
                g = -(1/2)*(x-self.means[i]) * self.invs[i] * (x-self.means[i]).T\
                    -(1/2) * np.log(np.abs(self.covs[i])) + np.log(self.args[i].shape[1]/self.total+1e-6)
            
                gs.append(g)

            ans = np.argmax(gs)
            
            return ans
        else:
            for arg in self.args:
                mean = np.mean(arg, axis=0)
                cov = np.cov(arg.T)
                inv = np.linalg.inv(cov)
                #print(inv)
                det = np.linalg.det(cov)
                if det == 0:
                    print("분류기를 만들 수 없음")
                #print(det)
                self.means.append(mean)
                self.covs.append(cov)
                self.invs.append(inv)
                self.dets.append(det)
            gs=[]
            for i in range(len(self.args)):
                g = -(1/2)*(x-self.means[i]) @ self.invs[i] @ (x-self.means[i]).T\
                    -(1/2) * np.log(self.dets[i]) + np.log(len(self.args[i])/self.total+1e-6)
                gs.append(g)
            ans = np.argmax(gs)
            return ans
        
class PCA:
    def __init__(self, train0, test0, train1, test1):
        self.train0 = train0
        self.train1 = train1
        self.test0 = test0
        self.test1 = test1
        mu_x0 = np.mean(train0, axis=0)
        mu_x1 = np.mean(train1, axis=0)

    def __call__(self, interval=2):
        
        train = np.concatenate((self.train0, self.train1), axis=0)
        cov = np.cov(train.T)
        self.eVal, self.eVec= np.linalg.eig(cov)
        self.eVec = self.eVec[self.eVal.argsort()[::-1]]
        trn0 = np.array((self.train0 @ self.eVec[:, :interval]), ndmin=2)
        trn1 = np.array((self.train1 @ self.eVec[:, :interval]), ndmin=2)
        tst0 = np.array((self.test0 @ self.eVec[:, :interval]), ndmin=2)
        tst1 = np.array((self.test1 @ self.eVec[:, :interval]), ndmin=2)
                
        return trn0, tst0, trn1, tst1
class LDA:
    def __init__(self, train0, test0, train1, test1):
        self.train0 = train0
        self.train1 = train1
        self.test0 = test0
        self.test1 = test1
        cov_x0 = np.cov(train0.T) 
        cov_x1 = np.cov(train1.T) 
        mu_x0 = np.mean(train0, axis=0)
        mu_x1 = np.mean(train1, axis=0)
        self.Sw = cov_x0 + cov_x1
        self.Sw_inv= np.linalg.inv(self.Sw)
        self.Sb = np.array(mu_x0 - mu_x1, ndmin=2).T @ \
                  np.array(mu_x0 - mu_x1, ndmin=2)

    def __call__(self, interval=2):      
        m = self.Sw_inv @ self.Sb
        self.eVal, self.eVec= np.linalg.eig(m)
        self.eVec = self.eVec[self.eVal.argsort()[::-1]]
        trn0 = np.array((self.train0 @ self.eVec[:, :interval]), ndmin=2)
        trn1 = np.array((self.train1 @ self.eVec[:, :interval]), ndmin=2)
        tst0 = np.array((self.test0 @ self.eVec[:, :interval]), ndmin=2)
        tst1 = np.array((self.test1 @ self.eVec[:, :interval]), ndmin=2)
        return trn0, tst0, trn1, tst1
        

if __name__ == "__main__":
    x = [1, 2, 3, 4]
    y = [5, 6, 7, 8]
    x = np.array(x, ndmin=2)
    y = np.array(y, ndmin=2)
    classifier = Classifier(x, y)
    ans = classifier([2])
    
    
    
            
    
    
