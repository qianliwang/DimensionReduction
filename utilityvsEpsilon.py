from sklearn.model_selection import StratifiedShuffleSplit;
from sklearn.preprocessing import StandardScaler;
from sklearn import preprocessing;

import numpy as np;
from numpy import linalg as LA;
import sys;
import os;

from pkg.svm import SVMModule;
from pkg.dimReduction import PCAModule;
from pkg.diffPrivDimReduction import DiffPrivPCAModule;
from pkg.diffPrivDimReduction import DPModule;

def DPPro(pureTrainingData,pureTestingData,l2Sensitivity,k,epsilon):
    #preprocessing.normalize(pureTrainingData, copy=False);
    #preprocessing.normalize(pureTestingData, copy=False);
    projMatrixLength = pureTrainingData.shape[1]*k;
    oneDimNormalSamples = np.random.normal(0, np.divide(1.0,k), projMatrixLength);
    projMatrix = np.reshape(oneDimNormalSamples,(pureTrainingData.shape[1],-1));
    delta = np.divide(1.0, pureTrainingData.shape[0]);
    noiseLength = pureTrainingData.shape[0]*k;
    oneDimNoise = DPModule.DiffPrivImpl.OneDimGaussian(epsilon,delta,noiseLength,l2Sensitivity=l2Sensitivity);
    noiseMatrix = np.reshape(oneDimNoise,(pureTrainingData.shape[0],-1));

    projTrainingData = np.dot(pureTrainingData,projMatrix);
    noisyProjTrainingData = projTrainingData + noiseMatrix;
    projTestingData = np.dot(pureTestingData,projMatrix);

    return noisyProjTrainingData,projTestingData;

def singleExp(xEpsilons,trainingData,testingData,largestReducedFeature,isLinearSVM):
    pureTrainingData = trainingData[:,1:];
    trainingLabel = trainingData[:,0];
    
    numOfTrainingSamples = trainingData.shape[0];
    
    pureTestingData = testingData[:,1:];
    testingLabel = testingData[:,0];
    
    scaler = StandardScaler(copy=False);
    #print pureTrainingData[0];
    scaler.fit(pureTrainingData);
    scaler.transform(pureTrainingData);
    #print pureTrainingData[0];
    
    #print pureTestingData[0];
    scaler.transform(pureTestingData);
    #print pureTestingData[0];
    
    pcaImpl = PCAModule.PCAImpl(pureTrainingData);
    pcaImpl.getPCs(largestReducedFeature);
    
    dpGaussianPCAImpl = DiffPrivPCAModule.DiffPrivPCAImpl(pureTrainingData);
    dpWishartPCAImpl = DiffPrivPCAModule.DiffPrivPCAImpl(pureTrainingData);
    
    delta = np.divide(1.0,numOfTrainingSamples);
    projTrainingData1 = pcaImpl.transform(pureTrainingData,largestReducedFeature);
    projTestingData1 = pcaImpl.transform(pureTestingData,largestReducedFeature);
    #print projTrainingData.shape;
    cprResult = [];
    print "non noise PCA SVM training";
    if isLinearSVM:
        result = SVMModule.SVMClf.linearSVM(projTrainingData1,trainingLabel,projTestingData1,testingLabel);
    else:
        result = SVMModule.SVMClf.rbfSVM(projTrainingData1,trainingLabel,projTestingData1,testingLabel);
    #cprResult[0][1] += result[0];
    #cprResult[0][2] += result[1];
    #cprResult[0][3] += result[2];
    pcaAccuracy = result[3];

    for k, targetEpsilon in np.ndenumerate(xEpsilons):
        #print pcaImpl.projMatrix[:,0];    
        print "epsilon: %.2f, delta: %f" % (targetEpsilon,delta);

        cprResult.append(targetEpsilon);
        cprResult.append(pcaAccuracy);

        isGaussianDist = True;
        dpGaussianPCAImpl.setEpsilonAndGamma(targetEpsilon,delta);
        dpGaussianPCAImpl.getDiffPrivPCs(isGaussianDist,largestReducedFeature);
        
        '''
        We don't need to project the data multiple times.
        '''

        """
        isGaussianDist = False;
        dpWishartPCAImpl.setEpsilonAndGamma(targetEpsilon,delta);
        dpWishartPCAImpl.getDiffPrivPCs(isGaussianDist,largestReducedFeature);
        
        cprResult[k][0] += targetEpsilon;
        if k>0:
            cprResult[k][1] += cprResult[0][1];
            cprResult[k][2] += cprResult[0][2];
            cprResult[k][3] += cprResult[0][3];

        """

        projTrainingData2 = dpGaussianPCAImpl.transform(pureTrainingData,largestReducedFeature);
        projTestingData2 = dpGaussianPCAImpl.transform(pureTestingData,largestReducedFeature);
        print "Gaussian-DPDPCA SVM training";
        
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData2,trainingLabel,projTestingData2,testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData2,trainingLabel,projTestingData2,testingLabel);
        #cprResult[k][4] += result[0];
        #cprResult[k][5] += result[1];
        #cprResult[k][6] += result[2];
        cprResult.append(result[3]);
        """
        projTrainingData3 = dpWishartPCAImpl.transform(pureTrainingData,largestReducedFeature);
        projTestingData3 = dpWishartPCAImpl.transform(pureTestingData,largestReducedFeature);
        #print projTestingData.shape;
        print "Wishart-DPPCA SVM training";
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData3,trainingLabel,projTestingData3,testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData3,trainingLabel,projTestingData3,testingLabel);
        cprResult[k][7] += result[0];
        cprResult[k][8] += result[1];
        cprResult[k][9] += result[2];
        
        """
        projTrainingData4, projTestingData4 = DPPro(pureTrainingData, pureTestingData, dpGaussianPCAImpl.L2Sensitivity,
                                                    largestReducedFeature, targetEpsilon);
        # print projTestingData.shape;
        print "DPPro SVM training";
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData4, trainingLabel, projTestingData4, testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData4, trainingLabel, projTestingData4, testingLabel);
        cprResult.append(result[3]);

    resultArray = np.asarray(cprResult);
    resultArray = np.reshape(resultArray,(len(xEpsilons),-1));
    return resultArray;


def doExp(datasetPath,varianceRatio,numOfRounds,isLinearSVM=True):
    if os.path.basename(datasetPath).endswith('npy'):
        data = np.load(datasetPath);
    else:
        data = np.loadtxt(datasetPath, delimiter=",");
    scaler = StandardScaler();
    data_std = scaler.fit_transform(data[:, 1:]);
    globalPCA = PCAModule.PCAImpl(data_std);
    numOfFeature = data.shape[1]-1;
    largestReducedFeature = globalPCA.getNumOfPCwithKPercentVariance(varianceRatio);
    print "%d/%d dimensions captures %.2f variance." % (largestReducedFeature,numOfFeature,varianceRatio);

    xEpsilons = np.arange(0.1,1.1,0.1);
    cprResult = None;
    rs = StratifiedShuffleSplit(n_splits=numOfRounds, test_size=.2, random_state=0);
    rs.get_n_splits(data[:,1:],data[:,0]);
    for train_index, test_index in rs.split(data[:,1:],data[:,0]):
        trainingData = data[train_index];
        testingData = data[test_index];
        tmpResult = singleExp(xEpsilons,trainingData,testingData,largestReducedFeature,isLinearSVM);
        if cprResult is None:
            cprResult = tmpResult;
        else:
            cprResult = np.concatenate((cprResult,tmpResult),axis=0);

    for result in cprResult:
        print ','.join(['%.3f' % num for num in result]);
    
    return cprResult;

if __name__ == "__main__":
    
    numOfRounds = 2;
    varianceRatio = 0.8;
    figSavedPath = "./log/";
    resultSavedPath = "./log/";
    isLinearSVM = False;
    if len(sys.argv) > 1:
        datasetPath = sys.argv[1];
        varianceRatio = 0.8;
        print "+++ using passed in arguments: %s" % (datasetPath);
        result = doExp(datasetPath,varianceRatio,numOfRounds,isLinearSVM=isLinearSVM);
        np.savetxt(resultSavedPath+"Epsilon_"+os.path.basename(datasetPath)+".output",result,delimiter=",",fmt='%1.3f');
    else:
        datasets = ['CNAE_2','Face_15','Amazon_5','diabetes','ionosphere'];
        for dataset in datasets:    
            print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
            datasetPath = "./input/"+dataset+"_prePCA";
            result = doExp(datasetPath,varianceRatio,numOfRounds,isLinearSVM=isLinearSVM);
            np.savetxt(resultSavedPath+"Epsilon_"+dataset+".output",result,delimiter=",",fmt='%1.3f');
