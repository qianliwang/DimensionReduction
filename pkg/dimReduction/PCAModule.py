import numpy as np;
from numpy import linalg as LA;

class PCAImpl(object):
    
    def __init__(self,scaledData):
        self.eigValues,self.projMatrix = self.__getPCs(scaledData);
        self.energies = self.__getEigValueEnergies(self.eigValues);
        
    def __getPCs(self,scaledData):
        covMatrix = np.dot(scaledData.T,scaledData);
        w, v = LA.eig(covMatrix);    
        # Sorting the eigenvalues in descending order.
        idx = np.absolute(w).argsort()[::-1];
        #print idx;
        sortedW = w[idx];
        #print sortedW;
        sortedV = v[:,idx];
        
        return sortedW,sortedV;
    
    def __getApproxEigval(self,covMatrix,r1):
        temp1 = np.dot(covMatrix,r1);
        v1 = np.dot(r1.T,temp1);
        v2 = np.dot(r1.T,r1);
        eigVal = np.divide(v1,v2);
        return eigVal;

    def genEigenvectors_power(self,covMatrix):
    #    r0 = np.random.rand(covMatrix.shape[0],1);
        epsilon = 0.01;
        eigVectors = [];
        k=0;
        while k<covMatrix.shape[0]:
            r0 = np.random.rand(covMatrix.shape[0],1);
            count=0;
            while True:
                r1 = np.dot(covMatrix, r0);
                # Get the second norm of r1;
                scale = LA.norm(r1,2);
                r1 = np.divide(r1,scale);
                #dist = LA.norm(r1-r0,2);
                # Note the formula to calculate the distance 
                eigVal = self.__getApproxEigval(covMatrix,r1);
                dist = LA.norm(np.dot(covMatrix,r1)-eigVal*r1,2);
                #print dist;
                
                if dist < epsilon:
                    #print count;
                    #print eigVal;
                    break;
                else:    
                    r0 = r1;
                    count = count + 1;
            #print (r1.dot(r1.T)); 
            eigVectors.append(r1);
            covMatrix = covMatrix - covMatrix.dot(r1.dot(r1.T));
            k = k+1;            
        return np.asarray(eigVectors).T;
    
    def __getEigValueEnergies(self,w):
        totalEnergy = np.sum(w);
        return [elem/totalEnergy for elem in w];
    
    def transform(self,scaledData,numOfComponents):
        if(numOfComponents>len(self.eigValues)):
            print "This PCA could only project data up to %d dimension." % len(self.eigValues);
        tmpNumOfComponents = len(self.eigValues) if numOfComponents>len(self.eigValues) else numOfComponents;
        tmpProjMatrix = self.projMatrix[:,0:tmpNumOfComponents];
        return np.dot(scaledData,tmpProjMatrix);