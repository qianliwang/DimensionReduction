import numpy as np;
from numpy import linalg as LA;
#import scipy;
#from scipy import sparse;
import scipy.sparse as sparse;
from scipy.sparse.linalg import svds;
from scipy.spatial.distance import pdist, squareform, cdist

def scipySvdSolver(covMatrix,topK):
    #u,s,v = sparse.linalg.svds(covMatrix, k=topK, tol=0.001);
    u,s,v = svds(covMatrix, k=topK, tol=0.001);
    return np.real(s),np.real(v.T);

def getApproxEigval(covMatrix,r1):
    temp1 = np.dot(covMatrix,r1);
    v1 = np.dot(r1.T,temp1);
    v2 = np.dot(r1.T,r1);
    eigVal = np.divide(v1,v2);
    return eigVal;

def genEigenvectors_power(covMatrix,topK,epsilon=0.01):
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
            eigVal = getApproxEigval(covMatrix,r1);
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
    print("converge rounds: %s" % convergeRounds);         
    return eigValues,eigVectors;

def svdSolver(meanCenteredData):
    '''
    Using Singular Value Decomposition to find the eigenvalues and principal components.
    '''
    U, s, V = LA.svd(meanCenteredData, full_matrices=False)
    
    eigValues = np.square(s);
    #print eigValues[:20];
    eigVectors = np.real(V.T);
    #print eigVectors[:20,1];
    return eigValues,eigVectors;
    
def evdSolver(covMatrix):
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

def scipyEvdSolver(covMatrix,topK):
    w,v = sparse.linalg.eigs(covMatrix, k=topK, tol=0.001);
    idx = np.absolute(w).argsort()[::-1];
    #print idx;
    sortedW = w[idx];
    #print sortedW;
    sortedV = v[:,idx];
    return np.real(sortedW),np.real(sortedV);

class PCAImpl(object):
    """
    Self-implemented Principal Component Analysis. Important notice: 
        1) The input is raw data, MxN format, M number of samples, N number of features
        2) EigenDecomposition is implemented.
        3) SingularValueDecomposition is implemented.
        4) Power Iteration is implemented.
    """    
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
            self.eigValues,self.projMatrix = svdSolver(self.centeredData);
        elif topK is not None:
            self.eigValues,self.projMatrix = scipySvdSolver(self.covMatrix,topK);
            #print "Power Iteration to find top %d principal components." % topK;
            #self.eigValues,self.projMatrix = self.genEigenvectors_power(self.covMatrix,topK);
        else:
            #print "Eigenvalue decomposition";
            self.eigValues,self.projMatrix = evdSolver(self.covMatrix);
    
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
            print("This PCA could only project data up to %d dimension." % len(self.eigValues));
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


class KernelPCAImpl(object):
    def __init__(self, scaledData):
        '''
        Tuning parameter, gamma is hard-coded.
        '''
        self.scaledData = scaledData;
        self.gamma = 15;
        self.eigValues, self.projMatrix = self.__getPCs(scaledData);
        self.normEigvectors = self.projMatrix / self.eigValues;

    def __getPCs(self, scaledData):
        '''
        Calculate the squared Euclidean distance of each pairs of sacledData, then make the distance in a symmetric matrix.
        '''
        distMatrix = cdist(scaledData, scaledData, 'sqeuclidean');

        kernelMatrix = np.exp(-self.gamma * distMatrix);

        IdMatrix = np.ones(kernelMatrix.shape) * (1.0 / len(kernelMatrix));
        # Centering the kernel matrix, which could be done in the raw data,
        # which is better, since I got a scale of the centerPara is much larger in magnitude.
        centerPara = np.dot(IdMatrix.dot(kernelMatrix), IdMatrix) - IdMatrix.dot(kernelMatrix) - kernelMatrix.dot(IdMatrix);
        centeredKernelMat = kernelMatrix + centerPara;

        return evdSolver(centeredKernelMat);


    def getProjedTrainingData(self, numOfComponents):
        return self.projMatrix[:, :numOfComponents];

    def transformNewData(self, newScaledData, numOfComponents):
        '''
        No need of the training kernel matrix, just need the eigenvectors of the training kernel matrix, due to the kernel trick.
        '''
        newDistMatrix = cdist(newScaledData, self.scaledData, 'sqeuclidean');

        newKernelMatrix = np.exp(-self.gamma * newDistMatrix);

        return newKernelMatrix.dot(self.normEigvectors[:, :numOfComponents]);

class LDAImpl(object):

    '''
    Self-implemented Linear Discriminant Analysis. Notice:
        1) The raw data is no needed to be centered.
        2) The data and corresponding labels are separated as input, they are concatenated together in the constructor. The
           reason is to align with PCA.
    '''    
    def __init__(self,scaledData,labels):
        self.numOfClass = 0;
        #self.labeledScaleData = np.hstack((np.reshape(labels,(-1,1)),scaledData));
        self.Sw,self.Sb = self.__getSwAndSb(scaledData,labels);
        self.SwInv = LA.inv(self.Sw);
        self.projMatrix = None;
        
    def getPCs(self,n_components):
        '''
        Self implemented LDA, the result is same with the implementation of scikit-learn.
        '''
        print("LDA Implementation.");
        withinBetweenMatrix = np.dot(self.SwInv,self.Sb);
        n_components = min(n_components,self.numOfClass-1);
        w,v = scipySvdSolver(withinBetweenMatrix,n_components);
        
        self.projMatrix = v;
            
    def __getSwAndSb(self,scaledData,labels):
        '''
        Compute the withinClass covariance and betweenClass covariance from the labelled data.
        '''
        # Using a dictionary to categorize the data.
        uniqueLabel = np.unique(labels);
        dim=scaledData.shape[1];
        # Initialize the numOfClass.
        self.numOfClass = len(uniqueLabel);

        overallMean = np.mean(scaledData,axis=0);
        
        S_W = np.zeros((dim,dim))
        S_B = np.zeros((dim,dim));
        #print(S_B.shape)
        for label in uniqueLabel:
            class_sc_mat = np.zeros((dim,dim));
            classIdx = np.where(labels==label);
            #print(classIdx);
            classData = scaledData[classIdx];
            classMean = np.mean(classData,axis=0);
            #print(classData);
            classCentered = classData-classMean;
            #print(classCentered)
            class_sc_mat += classCentered.T.dot(classCentered);
            S_W += class_sc_mat;
            #print(S_W);
            meanDiff = classMean - overallMean;
            #print(meanDiff);
            S_B += len(classIdx[0])*np.outer(meanDiff,meanDiff);
            
        """
        labelDict = {};
        for label in uniqueLabel:
            labelDict[label] = scaledData[labels==label];
            print(labelDict[label].shape);
            
        # Compute the withinClass covariance.
        aggreDict = {};
        tmpTotalMean = None;
        tmpTotalCount = 0;
        withinClassCov = None;
        
        for key,value in labelDict.items():
            tmpNdArray = np.asarray(value);
            if tmpTotalMean is None:
                tmpTotalMean = np.zeros(tmpNdArray.shape[1]);
            if withinClassCov is None:
                withinClassCov = np.zeros((tmpNdArray.shape[1],tmpNdArray.shape[1]));
            withinClassMean = np.mean(tmpNdArray,axis = 0);
            centeredNdArray = tmpNdArray-withinClassMean;
            withinClassCov = withinClassCov + np.dot(centeredNdArray.T,centeredNdArray);
            aggreDict[key] = [withinClassMean,tmpNdArray.shape[0]];
            tmpTotalMean = tmpTotalMean+np.sum(tmpNdArray,axis = 0);
            tmpTotalCount = tmpTotalCount + tmpNdArray.shape[0];
        totalMean = np.divide(tmpTotalMean,tmpTotalCount);
        covSize = len(tmpTotalMean);
        
        # Compute the betweenClass covariance.
        betweenClassCov = np.zeros((covSize,covSize));
        
        for key,value in aggreDict.items():
            withinClassMean = value[0];
            withinClassCount = value[1];
            tmpMeanDiff = withinClassMean - totalMean;
            tmpBetweenCov = np.outer(tmpMeanDiff,tmpMeanDiff.T);
            betweenClassCov = betweenClassCov + withinClassCount*tmpBetweenCov;

        print((S_B==betweenClassCov).all());
        return withinClassCov,betweenClassCov;
        """
        return S_W, S_B;
        
    def transform(self,scaledData,numOfComponents):
        if(self.projMatrix is None):
            self.getPCs();
        if(numOfComponents>(self.numOfClass-1)):
            print("This discrimination could only project data up to %d dimension due the number of classes." % (self.numOfClass-1));
        tmpNumOfComponents = self.numOfClass-1 if numOfComponents>(self.numOfClass-1) else numOfComponents;
        
        tmpProjMatrix = self.projMatrix[:,0:tmpNumOfComponents];
        return np.dot(scaledData,tmpProjMatrix);

class DCAImpl(LDAImpl):
    '''
    Self-implemented Discriminant Analysis, which is inherited from LDA. Comparint to LDA, DCA has two ridge parameters,
    rho and rho_prime, the scatter matrix is different from LDA.
    '''    
    def __init__(self,scaledData,labels,rho=0,rho_prime=0):
        
        LDAImpl.__init__(self,scaledData,labels);
        self.rho = rho;
        self.rho_prime = rho_prime;
        
    def setRhoAndRho_prime(self,rho,rho_prime):
        self.rho = rho;
        self.rho_prime = rho_prime;
    
    def getPCs(self,n_components):
        '''
        Self implemented DCA. Notice the calculation of S_prime, which is different from LDA's scatter matrix.
        '''
        print("DCA Implementation.");
        
        S_prime = self.Sw+self.Sb+(self.rho+self.rho_prime)*np.ones(self.Sw.shape);
        discriminantMatrix = np.dot(self.SwInv,S_prime);
        n_components = min(n_components,self.numOfClass-1);
        w,v = scipySvdSolver(discriminantMatrix,n_components);
        eigenEnergySum = np.sum(w);
        print("The first eigenvector of DCA takes %f variance." % (w[0]/eigenEnergySum));
        self.projMatrix = v;
