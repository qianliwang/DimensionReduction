import numpy as np;
from numpy import linalg as LA;
from invwishart import *;
from DimReduction import PCAImpl,evdSolver,scipyEvdSolver;
'''
Two differentially private noise to add into the covariance, one using Gaussian distribution, the other one is based on 
Wishart distribution, from two different papers respectively, mentioned in README.
'''
class DiffPrivImpl(object):
    @classmethod
    def SymmGaussian(cls,epsilon,delta,dimension,deltaF):
        standardDeviation = np.sqrt(2*np.log(1.25/delta))*deltaF/epsilon;
        #print "Gaussian Standard deviation is %f." % standardDeviation;
        noiseMatrix = np.random.normal(0, standardDeviation, (dimension,dimension));
        #Copy upper triangle to lower triangle in a matrix.
        i_lower = np.tril_indices(dimension, -1);
        noiseMatrix[i_lower] = noiseMatrix.T[i_lower];
        
        #print noiseMatrix;
        return noiseMatrix;
    
    @classmethod
    def SymmWishart(cls,epsilon,dimension):
        df = dimension+1;
        sigma = 1/epsilon*np.identity(dimension);
        #print sigma;
        wishart = wishartrand(df,sigma);
        #print wishart;
        return wishart;
    
    @classmethod
    def SymmWishart_withDelta(cls,epsilon,delta,dimension,sensitivity):
        df = dimension+int(np.floor(14.0/(epsilon*epsilon)*(2.0*np.log(4.0/delta))));
        sigma = (sensitivity*sensitivity)*np.identity(dimension);
        #print sigma;
        wishart = wishartrand(df,sigma);
        #print wishart;
        return wishart;

    @classmethod
    def OneDimGaussian(cls, epsilon, delta, size, mu=0, l2Sensitivity=1):
        standardDeviation = np.sqrt(2 * np.log(1.25 / delta)) * l2Sensitivity / epsilon;
        samples = np.random.normal(mu, standardDeviation, size);
        return samples;


'''
Differentially private PCA implementation, with both Gaussian noise and Wishart noise, details of the noise is in DPModule.py.
It is inherited from PCAImpl. In the paper, the data should be both centered and normalized, once normalized, the sensitivity 
is taken with the upper bound 1; here I just centered, then calculate the sensitivity from the centered data.
'''

class DiffPrivPCAImpl(PCAImpl):

    def __init__(self, rawData):

        PCAImpl.__init__(self, rawData);

        self.epsilon = 0;
        self.delta = 0;
        self.L2Sensitivity = self.__calcL2Sensitivity(self.centeredData);
        self.frobeniusSensitivity = self.L2Sensitivity * self.L2Sensitivity;
        # self.frobeniusSensitivity = self.__calcFrobeniusSensitivity(self.centeredData);

    def setEpsilonAndGamma(self, epsilon, delta):
        self.epsilon = epsilon;
        self.delta = delta;

    def getDiffPrivPCs(self, isGaussianNoise, topK=None, onlyEigvalues=False):

        if isGaussianNoise:
            noiseMatrix = DiffPrivImpl.SymmGaussian(self.epsilon, self.delta, self.covMatrix.shape[0],
                                                    self.frobeniusSensitivity);
        else:
            noiseMatrix = DiffPrivImpl.SymmWishart_withDelta(self.epsilon, self.delta, self.covMatrix.shape[0],
                                                             self.L2Sensitivity);

        # print wishart;
        noisyCovMatrix = self.covMatrix + noiseMatrix;
        if onlyEigvalues:
            sigValues = LA.svd(noisyCovMatrix, compute_uv=False);
            self.eigValues = np.square(sigValues);
        else:
            if self.centeredData.shape[1] < 500:
                # print "Eigenvalue decomposition";
                self.eigValues, self.projMatrix = evdSolver(noisyCovMatrix);
            else:
                self.eigValues, self.projMatrix = scipyEvdSolver(noisyCovMatrix, topK);
                # print "Power Iteration to find top %d principal components." % topK;
                # self.eigValues,self.projMatrix = PCAModule.PCAImpl.genEigenvectors_power(self,noisyCovMatrix,topK);

        # print self.projMatrix[:20,1];

    def __calcFrobeniusSensitivity(self, data):
        firstMaxSensitivity = 0;
        for i in range(0, len(data)):
            data_prime = np.delete(data, (i), axis=0);
            # print data.shape;
            # print data_prime.shape;
            diffMatrix = np.dot(data.T, data) - np.dot(data_prime.T, data_prime);
            tmpNorm = LA.norm(diffMatrix, 'fro');
            firstMaxSensitivity = max(firstMaxSensitivity, tmpNorm);
            # print tmpNorm;
        print "The 1st Frobinus sensitivity of the data is %f." % firstMaxSensitivity;
        secondMaxSensitivity = 0;
        for singleData in data:
            tmpOuterMatrix = np.outer(singleData, singleData);
            tmpFroNorm = LA.norm(tmpOuterMatrix, 'fro');
            secondMaxSensivity = max(secondMaxSensitivity, tmpFroNorm);
        print "The 2nd Frobinus sensitivity of data is %f." % secondMaxSensivity;

        return firstMaxSensitivity;

    def __calcL2Sensitivity(self, data):
        rowsNorm = LA.norm(data, axis=1);
        maxL2Norm = np.amax(rowsNorm);
        # print "The L2 sensitivity of the data is %f." % maxL2Norm;
        return maxL2Norm;

    def transform(self, rawData, numOfComponents):
        return PCAImpl.transform(self, rawData, numOfComponents);

