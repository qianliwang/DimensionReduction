import numpy as np;
from numpy import linalg as LA;
from ..dimReduction import DCAModule;
from DPModule import DiffPrivImpl;

class DiffPrivDCAImpl(DCAModule.DCAImpl):
    
    def __init__(self,scaledData,labels):
        
        DCAModule.DCAImpl.__init__(self,scaledData,labels);
        self.maxGaussianSensitivity = self.__calcSensitivity(self.labeledScaleData);
        self.epsilon = 0;
        self.delta = 0;
        
    def setEpsilonAndGamma(self,epsilon,delta):
        self.epsilon = epsilon;
        self.delta = delta; 
        self.__getDPDCAPCs();
    
    def getPCs(self):
        self.__getDPDCAPCs();
        
    def __getDPDCAPCs(self):
        
        S_prime = self.Sw+self.Sb+(self.rho+self.rho_prime)*np.ones(self.Sw.shape);
        
        noiseMatrix = DiffPrivImpl.SymmGaussian(self.epsilon,self.delta,len(self.Sw),self.maxGaussianSensitivity);
        noisyS_prime = S_prime + noiseMatrix;
        
        noiseDiscriminantMatrix = np.dot(self.SwInv,noisyS_prime);
        w,v = self.eigDecompose(noiseDiscriminantMatrix);
        eigenEnergySum = np.sum(w);
        print "The first eigenvector of Diff Private DCA takes %f variance." % (w[0]/eigenEnergySum);
        
        self.projMatrix = v[:,0:self.numOfClass-1];
    
    def __calcSensitivity(self,data):
        norms=[];
        for i in range(0,len(data)):
            data_prime = np.delete(data,(i),axis=0);
            Sw,Sb = self._LDAImpl__getSwAndSb(data);
            Sw_prime,Sb_prime = self._LDAImpl__getSwAndSb(data_prime);
            inverseSw = LA.inv(Sw);
            inverseSw_prime = LA.inv(Sw_prime);
            fA = np.dot(inverseSw,(Sw+Sb));
            fA_prime = np.dot(inverseSw_prime,(Sw_prime+Sb_prime));
            diffMatrix = fA-fA_prime;
            tmpNorm = LA.norm(diffMatrix,'fro');
            norms.append(tmpNorm);
            #print tmpNorm;
        maxSensitivity = np.amax(norms);
        print "The max sensitivity of DCA implementation is %f." % maxSensitivity;
        return maxSensitivity;