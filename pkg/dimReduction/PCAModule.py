import numpy as np;
from numpy import linalg as LA;
import scipy.sparse as sparse;
"""
Self-implemented Principal Component Analysis. Important notice: 
    1) The input is raw data, MxN format, M number of samples, N number of features
    2) EigenDecomposition is implemented.
    3) SingularValueDecomposition is implemented.
    4) Power Iteration is implemented.
"""
class PCAImpl(object):
    
    def __init__(self,rawData):
        self.mean = np.mean(rawData,axis=0);
        self.centeredData = rawData-self.mean; 
        #self.covMatrix = np.dot(self.centeredData.T,self.centeredData);
        self.covMatrix = np.cov(self.centeredData, rowvar=False);
        self.eigValues = None;
        self.projMatrix = None;
        
    def getPCs(self,topK=None):
        if self.centeredData.shape[1]<500:
            #print "Singular Value Decomposition"
            #self.eigValues,self.projMatrix = self.evdSolver(self.covMatrix);
            self.eigValues,self.projMatrix = self.svdSolver(self.centeredData);
        elif topK is not None:
            self.eigValues,self.projMatrix = self.scipyEvdSolver(self.covMatrix,topK);
            #print "Power Iteration to find top %d principal components." % topK;
            #self.eigValues,self.projMatrix = self.genEigenvectors_power(self.covMatrix,topK);
        else:
            #print "Eigenvalue decomposition";
            self.eigValues,self.projMatrix = self.evdSolver(self.covMatrix);
    def svdSolver(self,meanCenteredData):
        '''
        Using Singular Value Decomposition to find the eigenvalues and principal components.
        '''
        U, s, V = LA.svd(meanCenteredData, full_matrices=False)
        
        eigValues = np.square(s);
        #print eigValues[:20];
        eigVectors = np.real(V.T);
        #print eigVectors[:20,1];
        return eigValues,eigVectors;
    
    def evdSolver(self,covMatrix):
        '''
        1) Using the already computed covariance matrix.
        2) EigenDecomposition.
        3) Sort the eigenvalues in non-decreasing order and output corresponding eigenvectors with that order.  
        '''
        w, v = LA.eig(covMatrix);  
        # Sorting the eigenvalues in descending order.
        idx = np.absolute(w).argsort()[::-1];
        #print idx;
        sortedW = np.real(w[idx]);
        #print sortedW;
        sortedV = np.real(v[:,idx]);
        return sortedW,sortedV;
    
    def scipyEvdSolver(self,covMatrix,topK):
        w,v = sparse.linalg.eigs(covMatrix, k=topK, tol=0.001);
        idx = np.absolute(w).argsort()[::-1];
        #print idx;
        sortedW = w[idx];
        #print sortedW;
        sortedV = v[:,idx];
        return np.real(sortedW),np.real(sortedV);
    
    def __getApproxEigval(self,covMatrix,r1):
        temp1 = np.dot(covMatrix,r1);
        v1 = np.dot(r1.T,temp1);
        v2 = np.dot(r1.T,r1);
        eigVal = np.divide(v1,v2);
        return eigVal;

    def genEigenvectors_power(self,covMatrix,topK,epsilon=0.01):
        '''
        Compute the eigenvector with power iteration method, multiplying covariance with random vector, 
        converge threshold is setted through epsilon.
        '''
    #    r0 = np.random.rand(covMatrix.shape[0],1);
        eigValues = np.zeros(topK);
        eigVectors = None;
        convergeRounds = [];
        k=0;
        vecLength = covMatrix.shape[0];
        bound = max(1000,vecLength);
        while k<topK:
            r0 = np.random.rand(vecLength,1);
            count=0;
            while count<bound:
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
                    #print "No.%d eigenvalue: %f" % (k,eigVal);
                    break;
                else:    
                    r0 = r1;
                    count += 1;
            if eigVectors is None:
                eigVectors = r1;
            else:
                eigVectors = np.append(eigVectors,r1,axis=1);
            convergeRounds.append(count);
            np.put(eigValues,k,eigVal);
            covMatrix -= eigVal*np.outer(r1,r1);
            k += 1;
        print "converge rounds: %s" % convergeRounds;         
        return eigValues,eigVectors;
    
    def getEigValueEnergies(self):
        '''
        Once eigenvalues are computed, computing the percentage of each eigenvalue over the sum of eigenvalues. 
        '''
        if self.eigValues is None:
            sigValues = LA.svd(self.centeredData,compute_uv=False);
            self.eigValues = np.square(sigValues);
        absEigValues = np.absolute(self.eigValues);
        totalEnergy = np.sum(absEigValues);
        return [elem/totalEnergy for elem in absEigValues];
    
    def getNumOfPCwithKPercentVariance(self,varianceRatio):
        '''
        Once eigenvalues are computed, computing the num of principal components which explains k percent of whole variance,0<k<=1. 
        '''
        numOfDimension = 0;
        pcaEnergies = self.getEigValueEnergies();
        tmpSumEnergy = 0;
        for energy in pcaEnergies:
            tmpSumEnergy += energy;
            numOfDimension += 1;
            if tmpSumEnergy > varianceRatio:
                break;
        return numOfDimension;
    
    def transform(self,scaledData,numOfComponents):
        '''
        Given a set of centered data, reducing the data to specified dimensions. Notice that the data should also be 
        centered already.
        '''
        if(numOfComponents>len(self.eigValues)):
            print "This PCA could only project data up to %d dimension." % len(self.eigValues);
        tmpNumOfComponents = len(self.eigValues) if numOfComponents>len(self.eigValues) else numOfComponents;
        tmpProjMatrix = self.projMatrix[:,:tmpNumOfComponents];
        centeredScaledData = scaledData - self.mean;
        return np.dot(centeredScaledData,tmpProjMatrix);
    
    def reconstruct(self,reducedData):
        '''
        Reconstruct the data using the transpose of the projection matrix.
        '''
        reducedDim = reducedData.shape[1];
        reconData = reducedData.dot(self.projMatrix[:,:reducedDim].T) + self.mean;
        #reconData = reducedData.dot(self.projMatrix[:,0:reducedDim].T);
        return reconData; 