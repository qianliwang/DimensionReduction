import numpy as np;
from numpy import linalg as LA;
from LDAModule import LDAImpl;

class DCAImpl(LDAImpl):
    
    def __init__(self,scaledData,labels):
        LDAImpl.__init__(self,scaledData,labels);
        self.rho = 0;
        self.rho_prime = 0;
        
    def setRhoAndRho_prime(self,rho,rho_prime):
        self.rho = rho;
        self.rho_prime = rho_prime;
        self.__getDCAPCs();
    
    def getPCs(self):
        self.__getDCAPCs();
        
    def __getDCAPCs(self):
        '''
        Self implemented DCA.
        '''
        print "DCA Implementation.";
        
        S_prime = self.Sw+self.Sb+(self.rho+self.rho_prime)*np.ones(self.Sw.shape);
        discriminantMatrix = np.dot(self.SwInv,S_prime);
        w,v = self.eigDecompose(discriminantMatrix);
        eigenEnergySum = np.sum(w);
        print "The first eigenvector of DCA takes %f variance." % (w[0]/eigenEnergySum);
        
        self.projMatrix = v[:,0:self.numOfClass-1];
    