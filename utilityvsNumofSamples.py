from pkg.svm import SVMModule;
from pkg.dimReduction import PCAModule;
#from pkg.DPDPCA.DataOwner import DataOwnerImpl;
import numpy as np;
from numpy import linalg as LA;
from pkg.diffPrivDimReduction import invwishart;
from numpy.linalg import norm;
from sklearn.model_selection import ShuffleSplit;
from sklearn.model_selection import StratifiedShuffleSplit;
from pkg.diffPrivDimReduction.DPModule import DiffPrivImpl;
import sys;
import os;
from multiprocessing import Pool;
import scipy.sparse as sparse;
from sklearn import preprocessing;
from sklearn.preprocessing import StandardScaler;
from pkg.global_functions import globalFunction as gf;

def simulatePrivateLocalPCA(data,targetDimension,epsilon):

    C = data.T.dot(data);
    WishartNoiseMatrix = DiffPrivImpl.SymmWishart(epsilon,data.shape[1]);
    noisyC = C + WishartNoiseMatrix;

    noisyLeftSigVectors,noisySingularValues, noisyRightSigVectors = sparse.linalg.svds(noisyC, k=targetDimension, tol=0.001);
    noisySingularValues = np.real(noisySingularValues);
    noisyRightSigVectors = np.real(noisyRightSigVectors.T);
    S = np.diagflat(np.sqrt(noisySingularValues));
    P = np.dot(noisyRightSigVectors,S);
    return P;

def simulatePrivateGlobalPCA(data,numOfSamples,targetDimension,epsilon):
    numOfCopies = data.shape[0]/numOfSamples;
    dataOwnerGroups = np.array_split(data, numOfCopies);
    P = None;
    for singleDataOwnerCopy in dataOwnerGroups:
        k_prime = min(targetDimension,LA.matrix_rank(singleDataOwnerCopy));
        P_prime = simulatePrivateLocalPCA(singleDataOwnerCopy, k_prime, epsilon);
        if P is not None:
            k_prime = max(k_prime, LA.matrix_rank(P));
            tmpSummary = np.hstack((P_prime, P));
            P = simulatePrivateLocalPCA(tmpSummary.T, k_prime, epsilon);
        else:
            P = P_prime;
    #return P;
    return preprocessing.normalize(P, axis=0, copy=False);

def singleExp(xSamples,trainingData,testingData,topK,epsilon,isLinearSVM):

    pureTrainingData = trainingData[:,1:];
    trainingLabel = trainingData[:,0];

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

    preprocessing.normalize(pureTrainingData, copy=False);
    preprocessing.normalize(pureTestingData, copy=False);

    numOfFeature = trainingData.shape[1]-1;

    pcaImpl = PCAModule.PCAImpl(pureTrainingData);
    pcaImpl.getPCs(topK);
    
    '''
    To get a Wishart Noise projection matrix.
    '''
    WishartNoiseMatrix = DiffPrivImpl.SymmWishart(epsilon,numOfFeature);
    noisyCovMatrix = pcaImpl.covMatrix + WishartNoiseMatrix;

    noisyLeftSigVectors, noisyEigValues,noisyProjMatrix = sparse.linalg.svds(noisyCovMatrix, k=topK, tol=0.001);
    noisyProjMatrix = np.real(noisyProjMatrix.T);

    """
    projTrainingData2 = np.dot(pureTrainingData, noisyProjMatrix);
    projTestingData2 = np.dot(pureTestingData, noisyProjMatrix);

    print "DPDPCA %d" % topK;
    if isLinearSVM:
        result = SVMModule.SVMClf.linearSVM(projTrainingData2, trainingLabel, projTestingData2, testingLabel);
    else:
        result = SVMModule.SVMClf.rbfSVM(projTrainingData2, trainingLabel, projTestingData2, testingLabel);

    cprResult.append(result[2]);
    """

    cprResult = [];

    for k, numOfSamples in np.ndenumerate(xSamples):

        cprResult.append(numOfSamples);
        # Project the data using different projection matrix.
        projTrainingData1 = pcaImpl.transform(pureTrainingData, numOfSamples);
        projTestingData1 = pcaImpl.transform(pureTestingData, numOfSamples);
        print "Non-noise PCA %d" % numOfSamples;
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData1, trainingLabel, projTestingData1, testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData1, trainingLabel, projTestingData1, testingLabel);

        cprResult.append(result[2]);

        projTrainingData2 = np.dot(pureTrainingData, noisyProjMatrix[:, :numOfSamples]);
        projTestingData2 = np.dot(pureTestingData, noisyProjMatrix[:, :numOfSamples]);

        print "DPDPCA %d" % numOfSamples;
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData2, trainingLabel, projTestingData2, testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData2, trainingLabel, projTestingData2, testingLabel);

        cprResult.append(result[2]);

        pgProjMatrix = simulatePrivateGlobalPCA(pureTrainingData,numOfSamples,topK,epsilon);

        projTrainingData3 = np.dot(pureTrainingData,pgProjMatrix);
        projTestingData3 = np.dot(pureTestingData,pgProjMatrix);
        
        print "\nPrivateLocalPCA with %d data held by each data owner" % numOfSamples;
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData3,trainingLabel,projTestingData3,testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData3,trainingLabel,projTestingData3,testingLabel);
        cprResult.append(result[2]);

    resultArray = np.asarray(cprResult);
    resultArray = np.reshape(resultArray, (len(xSamples), -1));
    return resultArray;

def doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfPointsinXAxis, isLinearSVM=True):
    if os.path.basename(datasetPath).endswith('npy'):
        data = np.load(datasetPath);
    else:
        data = np.loadtxt(datasetPath, delimiter=",");
    numOfFeature = data.shape[1] - 1;
    scaler = StandardScaler();
    data_std = scaler.fit_transform(data[:, 1:]);
    globalPCA = PCAModule.PCAImpl(data_std);
    largestReducedFeature = globalPCA.getNumOfPCwithKPercentVariance(varianceRatio);


    print "%d/%d dimensions captures %.2f variance." % (largestReducedFeature,numOfFeature,varianceRatio);
    cprResult = None;


    #rs = StratifiedShuffleSplit(n_splits=numOfRounds, test_size=.2, random_state=0);
    #rs.get_n_splits(data[:,1:],data[:,0]);
    rs = ShuffleSplit(n_splits=numOfRounds, test_size=.2, random_state=0);
    rs.get_n_splits(data);
    #p = Pool(numOfRounds);
    for train_index, test_index in rs.split(data):
    #for train_index, test_index in rs.split(data[:,1:],data[:,0]):

        trainingData = data[train_index];
        testingData = data[test_index];
        print "number of training samples %d" % trainingData.shape[0];
        #tmpResult = p.apply_async(singleExp, (xDimensions,trainingData,testingData,topK,isLinearSVM));
        #cprResult += tmpResult.get();
        mostSamplesPerDataOwner = trainingData.shape[0] / 2;
        xSamples = np.arange(2, mostSamplesPerDataOwner, max(mostSamplesPerDataOwner/numOfPointsinXAxis, 1));
        print "number of samples be tested: %s" % xSamples;
        tmpResult = singleExp(xSamples, trainingData, testingData, largestReducedFeature, epsilon, isLinearSVM);
        if cprResult is None:
            cprResult = tmpResult;
        else:
            cprResult = np.concatenate((cprResult,tmpResult),axis=0);

    #p.close();
    #p.join();
    for result in cprResult:
        print ','.join(['%.3f' % num for num in result]);

    return cprResult;

if __name__ == "__main__":
    
    numOfRounds = 2;
    epsilon = 0.3;
    varianceRatio = 0.8
    #numOfSamples = 2;
    numOfPointsinXAxis = 20;
    figSavedPath = "./log/";
    resultSavedPath = "./log/";
    isLinearSVM = False;
    if len(sys.argv) > 1:
        datasetPath = sys.argv[1];
        print "+++ using passed in arguments: %s" % (datasetPath);
        result = doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfPointsinXAxis,isLinearSVM=isLinearSVM);
        np.savetxt(resultSavedPath+"dataOwner_"+os.path.basename(datasetPath)+".output",result,delimiter=",",fmt='%1.3f');
    else:
        datasets = ['Diabetes','CNAE_2','Face_15','p53_3000'];
        for dataset in datasets:
            print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
            datasetPath = "./input/"+dataset+"_prePCA";
            result = doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfPointsinXAxis,isLinearSVM=isLinearSVM);
            np.savetxt(resultSavedPath+"dataOwner_"+dataset+".output",result,delimiter=",",fmt='%1.3f');
