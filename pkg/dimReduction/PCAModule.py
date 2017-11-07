import numpy as np;
from numpy import linalg as LA;

"""
Self-implemented Principal Component Analysis. Important notice: 
    1) The input data should be already centered. (To do, should change to no centered, since mean vector should also be part 
    of the implementation, since mean vecotr is needed in the transform function).
    2) EigenDecomposition is adopted.
"""
class PCAImpl(object):
    
    def __init__(self,scaledData):
        self.data = scaledData; 
        self.mean = None;
        self.eigValues = None;
        self.projMatrix = None;
          
    def getPCs(self):
        '''
        1) Compute the covariance matrix.
        2) EigenDecomposition.
        3) Sort the eigenvalues in non-decreasing order and output corresponding eigenvectors with that order.  
        '''
        self.mean = np.mean(self.data,axis=0);
        meanCenteredData = self.data - self.mean;
        covMatrix = np.dot(meanCenteredData.T,meanCenteredData);
        w, v = LA.eig(covMatrix);    
        # Sorting the eigenvalues in descending order.
        idx = np.absolute(w).argsort()[::-1];
        #print idx;
        sortedW = w[idx];
        #print sortedW;
        realV = np.real(v);
        sortedV = realV[:,idx];
        
        self.eigValues = sortedW;
        self.projMatrix = sortedV;
    
    def __getApproxEigval(self,covMatrix,r1):
        temp1 = np.dot(covMatrix,r1);
        v1 = np.dot(r1.T,temp1);
        v2 = np.dot(r1.T,r1);
        eigVal = np.divide(v1,v2);
        return eigVal;

    def genEigenvectors_power(self,covMatrix):
        '''
        Compute the eigenvector with power iteration method, multiplying covariance with random vector, 
        converge threshold is setted through epsilon.
        '''
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
    
    def getEigValueEnergies(self):
        '''
        Once eigenvalues are computed, computing the percentage of each eigenvalue over the sum of eigenvalues. 
        '''
        absEigValues = np.absolute(self.eigValues);
        totalEnergy = np.sum(absEigValues);
        return [elem/totalEnergy for elem in absEigValues];
    
    def transform(self,scaledData,numOfComponents):
        '''
        Given a set of centered data, reduding the data to specified dimensions. Notice that the data should also be 
        centered already.
        '''
        if(numOfComponents>len(self.eigValues)):
            print "This PCA could only project data up to %d dimension." % len(self.eigValues);
        tmpNumOfComponents = len(self.eigValues) if numOfComponents>len(self.eigValues) else numOfComponents;
        tmpProjMatrix = self.projMatrix[:,0:tmpNumOfComponents];
        centeredScaledData = scaledData - self.mean;
        return np.dot(centeredScaledData,tmpProjMatrix);
