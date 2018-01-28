from pkg.svm import SVMModule;
from pkg.dimReduction import PCAModule;
from pkg.diffPrivDimReduction import DiffPrivPCAModule;
from pkg.diffPrivDimReduction import DPModule;
import numpy as np;
from sklearn.model_selection import ShuffleSplit;
from sklearn.model_selection import StratifiedShuffleSplit;
import sys;
import os;
from sklearn.preprocessing import StandardScaler;

def DPPro(pureTrainingData,pureTestingData,l2Sensitivity,k,epsilon):

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

def singleExp(xDimensions,trainingData,testingData,largestReducedFeature,epsilon,isLinearSVM):
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

    cprResult = np.zeros((len(xDimensions),5));
    pcaImpl = PCAModule.PCAImpl(pureTrainingData);
    
    pcaImpl.getPCs(largestReducedFeature);
    numOfTrainingSamples = trainingData.shape[0];
    
    delta = np.divide(1.0,numOfTrainingSamples);
    print "epsilon: %.2f, delta: %f" % (epsilon,delta);
    
    isGaussianDist = True;
    dpGaussianPCAImpl = DiffPrivPCAModule.DiffPrivPCAImpl(pureTrainingData);
    dpGaussianPCAImpl.setEpsilonAndGamma(epsilon,delta);
    dpGaussianPCAImpl.getDiffPrivPCs(isGaussianDist,largestReducedFeature);
    
    isGaussianDist = False;
    dpWishartPCAImpl = DiffPrivPCAModule.DiffPrivPCAImpl(pureTrainingData);
    dpWishartPCAImpl.setEpsilonAndGamma(epsilon,delta);
    dpWishartPCAImpl.getDiffPrivPCs(isGaussianDist,largestReducedFeature);
    
    for k, targetDimension in np.ndenumerate(xDimensions):    
        #print pcaImpl.projMatrix[:,0];
        #result = SVMModule.SVMClf.rbfSVM(pureTrainingData,trainingLabel,pureTestingData,testingLabel);
        #print k;
        cprResult[k][0] += targetDimension;
        projTrainingData1 = pcaImpl.transform(pureTrainingData,targetDimension);
        projTestingData1 = pcaImpl.transform(pureTestingData,targetDimension);
        print "Non-noise PCA %d" % targetDimension;
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData1,trainingLabel,projTestingData1,testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData1,trainingLabel,projTestingData1,testingLabel);
        
        cprResult[k][1] += result[2];
        
        isGaussianDist = True;
        #dpGaussianPCAImpl.getDiffPrivPCs(isGaussianDist);
        projTrainingData2 = dpGaussianPCAImpl.transform(pureTrainingData,targetDimension);
        projTestingData2 = dpGaussianPCAImpl.transform(pureTestingData,targetDimension);
        #print projTestingData.shape;
        print "Gaussian-noise PCA %d" % targetDimension;
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData2,trainingLabel,projTestingData2,testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData2,trainingLabel,projTestingData2,testingLabel);
        cprResult[k][2] += result[2];
        
        isGaussianDist = False;
        #dpWishartPCAImpl.getDiffPrivPCs(isGaussianDist);
        projTrainingData3 = dpWishartPCAImpl.transform(pureTrainingData,targetDimension);
        projTestingData3 = dpWishartPCAImpl.transform(pureTestingData,targetDimension);
        #print projTestingData.shape;
        print "Wishart-noise PCA %d" % targetDimension;
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData3,trainingLabel,projTestingData3,testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData3,trainingLabel,projTestingData3,testingLabel);
        cprResult[k][3] += result[2];

        projTrainingData4,projTestingData4 = DPPro(pureTrainingData,pureTestingData,dpGaussianPCAImpl.L2Sensitivity, targetDimension, epsilon);
        # print projTestingData.shape;
        print "DPPro %d" % targetDimension;
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData4, trainingLabel, projTestingData4, testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData4, trainingLabel, projTestingData4, testingLabel);
        cprResult[k][4] += result[2];

        """
        for result in cprResult:
            print "%f,%f,%f" % (result[0],result[1],result[2]);
        """
    return cprResult;

def doExp_unbalanced(datasetPath,epsilon,varianceRatio,numOfRounds,numOfDimensions,isLinearSVM=True):
    if os.path.basename(datasetPath).endswith('npy'):
        data = np.load(datasetPath);
    else:
        data = np.loadtxt(datasetPath, delimiter=",");

    globalPCA = PCAModule.PCAImpl(data[:, 1:]);

    numOfFeature = data.shape[1] - 1;
    largestReducedFeature = globalPCA.getNumOfPCwithKPercentVariance(varianceRatio);
    print "%d/%d dimensions captures %.2f variance." % (largestReducedFeature, numOfFeature, varianceRatio);
    xDimensions = None;
    if numOfDimensions > numOfFeature:
        xDimensions = np.arange(1, numOfFeature);
        largestReducedFeature = numOfFeature;
    else:
        xDimensions = np.arange(1, largestReducedFeature, max(largestReducedFeature / numOfDimensions, 1));
    posiData = data[np.where(data[:,0]==1)];
    negData = data[np.where(data[:,0]==-1)];
    splitRatio = 0.8;
    numPosSamples = int(posiData.shape[0]*splitRatio);
    numNegSamples = int(negData.shape[0]*splitRatio);
    cprResult = None;
    #print posiData.shape;
    #print negData.shape;
    for i in range(numOfRounds):
        np.random.shuffle(posiData);
        np.random.shuffle(negData);
        trainingData = np.concatenate((posiData[:numPosSamples],negData[:numNegSamples]),axis=0);
        testingData = np.concatenate((posiData[numPosSamples:],negData[numNegSamples:]),axis=0);
        #print trainingData.shape;
        #print testingData.shape;
        tmpResult = singleExp(xDimensions, trainingData, testingData, largestReducedFeature, epsilon,isLinearSVM);
        if cprResult is None:
            cprResult = tmpResult;
        else:
            cprResult = np.concatenate((cprResult,tmpResult),axis=0);
    #avgCprResult = cprResult/numOfRounds;
    avgCprResult = cprResult;
    for result in avgCprResult:
        print "%d,%.3f,%.3f,%.3f,%.3f" % (result[0],result[1],result[2],result[3],result[4]);
    return avgCprResult;

def doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfDimensions,isLinearSVM=True):
    if os.path.basename(datasetPath).endswith('npy'):
        data = np.load(datasetPath);
    else:
        data = np.loadtxt(datasetPath, delimiter=",");
    scaler = StandardScaler();
    data_std = scaler.fit_transform(data[:,1:]);
    globalPCA = PCAModule.PCAImpl(data_std);

    numOfFeature = data.shape[1]-1;
    largestReducedFeature = globalPCA.getNumOfPCwithKPercentVariance(varianceRatio);
    print "%d/%d dimensions captures %.2f variance." % (largestReducedFeature,numOfFeature,varianceRatio);
    xDimensions = None;
    if numOfDimensions > numOfFeature:
        xDimensions = np.arange(1,numOfFeature);
        largestReducedFeature=numOfFeature;
    else:
        xDimensions = np.arange(1,largestReducedFeature,max(largestReducedFeature/numOfDimensions,1));
    
    #cprResult = np.zeros((len(xDimensions),4));
    cprResult = None;
    rs = StratifiedShuffleSplit(n_splits=numOfRounds, test_size=.2, random_state=0);
    rs.get_n_splits(data[:,1:],data[:,0]);

    for train_index, test_index in rs.split(data[:,1:],data[:,0]):
        trainingData = data[train_index];
        testingData = data[test_index];
        
        tmpResult = singleExp(xDimensions, trainingData, testingData, largestReducedFeature, epsilon,isLinearSVM);
        if cprResult is None:
            cprResult = tmpResult;
        else:
            cprResult = np.concatenate((cprResult,tmpResult),axis=0);
    #avgCprResult = cprResult/numOfRounds;
    avgCprResult = cprResult;
    for result in avgCprResult:
        print "%d,%.3f,%.3f,%.3f,%.3f" % (result[0],result[1],result[2],result[3],result[4]);
    return avgCprResult;

if __name__ == "__main__":
    #datasets = ['diabetes','german','ionosphere'];
    numOfRounds = 4;
    figSavedPath = "./log/";
    resultSavedPath = "./log/";
    numOfDimensions = 30;
    epsilon = 0.3;
    varianceRatio = 0.9;
    isLinearSVM = False;
    if len(sys.argv) > 1:
        datasetPath = sys.argv[1];
        print "+++ using passed in arguments: %s" % (datasetPath);
        #result = doExp_unbalanced(datasetPath,epsilon,varianceRatio,numOfRounds,numOfDimensions,isLinearSVM=isLinearSVM);
        result = doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfDimensions,isLinearSVM=isLinearSVM);
        np.savetxt(resultSavedPath+"numPC_"+os.path.basename(datasetPath)+".output",result,delimiter=",",fmt='%1.3f');
    else:
        datasets = ['diabetes','CNAE_2','CNAE_5','CNAE_7','face2','Amazon_3','madelon'];
        #datasets = ['diabetes','Amazon_2','Australian','german','ionosphere'];
        for dataset in datasets:
            print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
            datasetPath = "./input/"+dataset+"_prePCA";
            result = doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfDimensions,isLinearSVM=isLinearSVM);
            np.savetxt(resultSavedPath+"numPC_"+dataset+".output",result,delimiter=",",fmt='%1.3f');
