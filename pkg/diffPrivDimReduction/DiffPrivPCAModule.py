import numpy as np;
from numpy import linalg as LA;
from ..dimReduction import PCAModule;
from DPModule import DiffPrivImpl;

class DiffPrivPCAImpl(PCAModule.PCAImpl):
    
    def __init__(self,scaledData):
        
        PCAModule.PCAImpl.__init__(self,scaledData);
        
        self.maxGaussianSensitivity = self.__calcGaussianSensitivity();
        self.maxWishartSensitivity = self.__calcWishartSensitivity();
        self.epsilon = 0;
        self.delta = 0;
        
    def setEpsilonAndGamma(self,epsilon,delta):
        self.epsilon = epsilon;
        self.delta = delta; 
        
    def getDiffPrivPCs(self,isGaussianNoise):
        
        covMatrix = np.dot(self.data.T,self.data);
        
        if isGaussianNoise:
            noiseMatrix = DiffPrivImpl.SymmGaussian(self.epsilon,self.delta,len(covMatrix),self.maxGaussianSensitivity);
        else:
            noiseMatrix = DiffPrivImpl.SymmWishart_withDelta(self.epsilon,self.delta,len(covMatrix),self.maxWishartSensitivity);
            
        #print wishart;
        noisyCovMatrix = covMatrix+noiseMatrix;
        w, v = LA.eig(noisyCovMatrix);    
        # Sorting the eigenvalues in descending order.
        idx = np.absolute(w).argsort()[::-1];
        #print idx;
        sortedW = w[idx];
        self.eigValues = sortedW;
        #print sortedW;
        sortedV = v[:,idx];
        self.projMatrix = sortedV;
        
    def __calcGaussianSensitivity(self):
        norms = [];
        for i in range(0,len(self.data)):
            data_prime = np.delete(self.data,(i),axis=0);
            #print data.shape;
            #print data_prime.shape;
            diffMatrix = np.dot(self.data.T,self.data)-np.dot(data_prime.T,data_prime);
            tmpNorm = LA.norm(diffMatrix,'fro');
            norms.append(tmpNorm);
            #print tmpNorm;
        maxSensitivity = np.amax(norms);
        print "The Gaussian sensitivity of PCA implementation is %f." % maxSensitivity;
        return maxSensitivity;
    
    def __calcWishartSensitivity(self):
        maxL2Norm = 0;
        tmpL2Norm = 0;
        for i in range(0,len(self.data)):
            tmpL2Norm = LA.norm(self.data[i],2);
            if tmpL2Norm > maxL2Norm:
                maxL2Norm = tmpL2Norm;
        print "The Wishart sensitivity of PCA implementation is %f." % maxL2Norm;
        return maxL2Norm;
    