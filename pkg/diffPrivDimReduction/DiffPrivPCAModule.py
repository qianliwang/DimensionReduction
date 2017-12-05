import numpy as np;
from numpy import linalg as LA;
from ..dimReduction import PCAModule;
from DPModule import DiffPrivImpl;

'''
Differentially private PCA implementation, with both Gaussian noise and Wishart noise, details of the noise is in DPModule.py.
It is inherited from PCAImpl. In the paper, the data should be both centered and normalized, once normalized, the sensitivity 
is taken with the upper bound 1; here I just centered, then calculate the sensitivity from the centered data.
'''
class DiffPrivPCAImpl(PCAModule.PCAImpl):
    
    def __init__(self,rawData):
        
        PCAModule.PCAImpl.__init__(self,rawData);
        
        self.epsilon = 0;
        self.delta = 0;
        self.L2Sensitivity = self.__calcL2Sensitivity(self.centeredData);
        #self.frobeniusSensitivity = self.__calcFrobeniusSensitivity(self.centeredData);
        
    def setEpsilonAndGamma(self,epsilon,delta):
        self.epsilon = epsilon;
        self.delta = delta; 
        
    def getDiffPrivPCs(self,isGaussianNoise,topK=None):
        
        if isGaussianNoise:
            noiseMatrix = DiffPrivImpl.SymmGaussian(self.epsilon,self.delta,self.covMatrix.shape[0],self.L2Sensitivity);
        else:
            noiseMatrix = DiffPrivImpl.SymmWishart_withDelta(self.epsilon,self.delta,self.covMatrix.shape[0],self.L2Sensitivity);
            
        #print wishart;
        noisyCovMatrix = self.covMatrix+noiseMatrix;
        if self.centeredData.shape[1]<500:
            #print "Eigenvalue decomposition";
            self.eigValues,self.projMatrix = self.evdSolver(noisyCovMatrix);
        else:
            self.eigValues,self.projMatrix = PCAModule.PCAImpl.scipyEvdSolver(self,noisyCovMatrix,topK);
            #print "Power Iteration to find top %d principal components." % topK;
            #self.eigValues,self.projMatrix = PCAModule.PCAImpl.genEigenvectors_power(self,noisyCovMatrix,topK); 
         
        #print self.projMatrix[:20,1];
        
    def __calcFrobeniusSensitivity(self,data):
        firstMaxSensitivity = 0;
        for i in range(0,len(data)):
            data_prime = np.delete(data,(i),axis=0);
            #print data.shape;
            #print data_prime.shape;
            diffMatrix = np.dot(data.T,data)-np.dot(data_prime.T,data_prime);
            tmpNorm = LA.norm(diffMatrix,'fro');
            firstMaxSensitivity = max(firstMaxSensitivity,tmpNorm);
            #print tmpNorm;
        print "The 1st Frobinus sensitivity of the data is %f." % firstMaxSensitivity;
        secondMaxSensitivity = 0;
        for singleData in data:
            tmpOuterMatrix = np.outer(singleData,singleData);
            tmpFroNorm = LA.norm(tmpOuterMatrix,'fro');
            secondMaxSensivity = max(secondMaxSensitivity,tmpFroNorm);
        print "The 2nd Frobinus sensitivity of data is %f." % secondMaxSensivity;
        
        return firstMaxSensitivity;
    
    def __calcL2Sensitivity(self,data):
        rowsNorm = LA.norm(data, axis=1);
        maxL2Norm = np.amax(rowsNorm);
        #print "The L2 sensitivity of the data is %f." % maxL2Norm;
        return maxL2Norm;
    
    def transform(self,rawData,numOfComponents):
        return PCAModule.PCAImpl.transform(self,rawData,numOfComponents);