import numpy as np;
from numpy import linalg as LA;
from ..dimReduction import PCAModule;
from DPModule import DiffPrivImpl;

class DiffPrivPCAImpl(PCAModule.PCAImpl):
    
    def __init__(self,scaledData):
        
        isGaussianNoise = False;
        if isGaussianNoise:
            self.maxGaussianSensitivity = self.__calcSensitivity(scaledData);
        else:
            self.maxGaussianSensitivity = 0;
        self.eigValues,self.projMatrix = self.__getDiffPrivPCs(scaledData,isGaussianNoise,self.maxGaussianSensitivity);
        self.energies = self._PCAImpl__getEigValueEnergies(self.eigValues);
    
    def __getDiffPrivPCs(self,scaledData,isGaussianNoise,maxGaussianSensitivity):
        
        covMatrix = np.cov(scaledData.T);
        
        if isGaussianNoise:
            noiseMatrix = DiffPrivImpl.SymmGaussian(0.1,0.01,len(covMatrix),maxGaussianSensitivity);
        else:
            noiseMatrix = DiffPrivImpl.SymmWishart(0.1,len(covMatrix));
            
        
        #print wishart;
        noisyCovMatrix = covMatrix+noiseMatrix;
        w, v = LA.eig(noisyCovMatrix);    
        # Sorting the eigenvalues in descending order.
        idx = np.absolute(w).argsort()[::-1];
        #print idx;
        sortedW = w[idx];
        #print sortedW;
        sortedV = v[:,idx];
        return sortedW,sortedV;
    
    def __calcSensitivity(self,data):
        norms = [];
        for i in range(0,len(data)):
            data_prime = np.delete(data,(i),axis=0);
            #print data.shape;
            #print data_prime.shape;
            diffMatrix = np.dot(data.T,data)-np.dot(data_prime.T,data_prime);
            tmpNorm = LA.norm(diffMatrix,'fro');
            norms.append(tmpNorm);
            #print tmpNorm;
        maxSensitivity = np.amax(norms);
        print "The max sensitivity of PCA implementation is %f." % maxSensitivity;
        return maxSensitivity;