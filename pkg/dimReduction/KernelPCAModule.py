import numpy as np;
from numpy import linalg as LA;
from scipy.spatial.distance import pdist, squareform, cdist

'''
RBF kernel PCA implementation, may need to change the name to give an accurate description. Notice:
1) The input data should be centered, the implementation doesn't have a kernel matrix centralization.
'''
class KernelPCAImpl(object):
    
    def __init__(self,scaledData):
        '''
        Tuning parameter, gamma is hard-coded.
        '''
        self.scaledData = scaledData;
        self.gamma = 15;
        self.eigValues,self.projMatrix = self.__getPCs(scaledData);
        self.normEigvectors = self.projMatrix/self.eigValues;
        
    def __getPCs(self,scaledData):
        '''
        Calculate the squared Euclidean distance of each pairs of sacledData, then make the distance in a symmetric matrix.
        '''
        distMatrix = cdist(scaledData,scaledData,'sqeuclidean');
        
        kernelMatrix = np.exp(-self.gamma * distMatrix);
        
        IdMatrix = np.ones(kernelMatrix.shape)*(1.0/len(kernelMatrix));
        # Centering the kernel matrix, which could be done in the raw data,
        # which is better, since I got a scale of the centerPara is much larger in magnitude.
        centerPara = np.dot(IdMatrix.dot(kernelMatrix),IdMatrix) - IdMatrix.dot(kernelMatrix) - kernelMatrix.dot(IdMatrix);
        centeredKernelMat = kernelMatrix + centerPara;

        return self.eigDecompose(centeredKernelMat);
    
    def eigDecompose(self,covMatrix):
        w,v = LA.eig(covMatrix);
        idx = np.absolute(w).argsort()[::-1];
        #print idx;
        sortedW = w[idx];
        sortedV = v[:,idx];
        #print sortedW;
        #print sortedV;
        return sortedW,np.real(sortedV);
    
    def getProjedTrainingData(self,numOfComponents):
        return self.projMatrix[:,:numOfComponents];
    
    def transformNewData(self,newScaledData,numOfComponents):
        '''
        No need of the training kernel matrix, just need the eigenvectors of the training kernel matrix, due to the kernel trick.
        '''
        newDistMatrix = cdist(newScaledData, self.scaledData, 'sqeuclidean');
        
        newKernelMatrix = np.exp(-self.gamma * newDistMatrix);
        
        return newKernelMatrix.dot(self.normEigvectors[:,:numOfComponents]);
