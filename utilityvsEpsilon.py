from pkg.svm import SVMModule;
from pkg.dimReduction import PCAModule;
from pkg.diffPrivDimReduction import DiffPrivPCAModule;
import numpy as np;
from numpy import linalg as LA;
from sklearn.model_selection import ShuffleSplit;
from sklearn.model_selection import StratifiedShuffleSplit;
import sys;
import os;
from sklearn.preprocessing import StandardScaler;


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
    cprResult = np.zeros((len(xEpsilons),10)); 
    print "non noise PCA SVM training";
    if isLinearSVM:
        result = SVMModule.SVMClf.linearSVM(projTrainingData1,trainingLabel,projTestingData1,testingLabel);
    else:
        result = SVMModule.SVMClf.rbfSVM(projTrainingData1,trainingLabel,projTestingData1,testingLabel);
    cprResult[0][1] += result[0];
    cprResult[0][2] += result[1];
    cprResult[0][3] += result[2];
    for k, targetEpsilon in np.ndenumerate(xEpsilons):
        #print pcaImpl.projMatrix[:,0];    
        print "epsilon: %.2f, delta: %f" % (targetEpsilon,delta);
                
        isGaussianDist = True;
        dpGaussianPCAImpl.setEpsilonAndGamma(targetEpsilon,delta);
        dpGaussianPCAImpl.getDiffPrivPCs(isGaussianDist,largestReducedFeature);
        
        isGaussianDist = False;
        dpWishartPCAImpl.setEpsilonAndGamma(targetEpsilon,delta);
        dpWishartPCAImpl.getDiffPrivPCs(isGaussianDist,largestReducedFeature);
        '''
        We don't need to project the data multiple times.
        '''

        cprResult[k][0] += targetEpsilon;
        if k>0:
            cprResult[k][1] += cprResult[0][1];
            cprResult[k][2] += cprResult[0][2];
            cprResult[k][3] += cprResult[0][3];

        projTrainingData2 = dpGaussianPCAImpl.transform(pureTrainingData,largestReducedFeature);
        projTestingData2 = dpGaussianPCAImpl.transform(pureTestingData,largestReducedFeature);
        print "Gaussian-DPDPCA SVM training";
        
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData2,trainingLabel,projTestingData2,testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData2,trainingLabel,projTestingData2,testingLabel);
        cprResult[k][4] += result[0];
        cprResult[k][5] += result[1];
        cprResult[k][6] += result[2];
        
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
        
    return cprResult;


def doExp_unbalanced(datasetPath, varianceRatio, numOfRounds, isLinearSVM=True):
    if os.path.basename(datasetPath).endswith('npy'):
        data = np.load(datasetPath);
    else:
        data = np.loadtxt(datasetPath, delimiter=",");

    globalPCA = PCAModule.PCAImpl(data[:, 1:]);

    numOfFeature = data.shape[1] - 1;
    largestReducedFeature = globalPCA.getNumOfPCwithKPercentVariance(varianceRatio);
    print "%d/%d dimensions captures %.2f variance." % (largestReducedFeature, numOfFeature, varianceRatio);
    xEpsilons = np.arange(0.1, 1.0, 0.1);
    cprResult = None;

    posiData = data[np.where(data[:, 0] == 1)];
    negData = data[np.where(data[:, 0] == -1)];
    splitRatio = 0.8;
    numPosSamples = int(posiData.shape[0] * splitRatio);
    numNegSamples = int(negData.shape[0] * splitRatio);
    cprResult = None;
    # print posiData.shape;
    # print negData.shape;
    for i in range(numOfRounds):
        np.random.shuffle(posiData);
        np.random.shuffle(negData);
        trainingData = np.concatenate((posiData[:numPosSamples], negData[:numNegSamples]), axis=0);
        testingData = np.concatenate((posiData[numPosSamples:], negData[numNegSamples:]), axis=0);
        # print trainingData.shape;
        # print testingData.shape;
        tmpResult = singleExp(xEpsilons, trainingData, testingData, largestReducedFeature, isLinearSVM);
        if cprResult is None:
            cprResult = tmpResult;
        else:
            cprResult = np.concatenate((cprResult, tmpResult), axis=0);
    # avgCprResult = cprResult/numOfRounds;
    avgCprResult = cprResult;
    for result in avgCprResult:
        print ','.join(['%.3f' % num for num in result]);
    return avgCprResult;

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

    xEpsilons = np.arange(0.1,1.0,0.1);
    cprResult = None;
    #cprResult = np.zeros((len(xEpsilons),10));
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
    # To record the mean and standard deviation.
    avgResult = cprResult;
    #avgResult = cprResult/numOfRounds;
    for result in avgResult:
        print ','.join(['%.3f' % num for num in result]);
    
    return avgResult;

def normByRow(data):
    for i in range(data.shape[0]):
        rowNorm = LA.norm(data[i,:], ord=2);
        data[i,:] = data[i,:]/rowNorm;
    return data;
  
if __name__ == "__main__":
    
    numOfRounds = 2;
    varianceRatio = 0.9;
    figSavedPath = "./log/";
    resultSavedPath = "./log/";
    isLinearSVM = False;
    if len(sys.argv) > 1:
        datasetPath = sys.argv[1];
        varianceRatio = 0.8;
        print "+++ using passed in arguments: %s" % (datasetPath);
        #result = doExp_unbalanced(datasetPath,varianceRatio,numOfRounds,isLinearSVM=isLinearSVM);
        result = doExp(datasetPath,varianceRatio,numOfRounds,isLinearSVM=isLinearSVM);
        np.savetxt(resultSavedPath+"Epsilon_"+os.path.basename(datasetPath)+".output",result,delimiter=",",fmt='%1.3f');
    else:
        datasets = ['B11','Face_15','CNAE_2','Amazon_5','diabetes','ionosphere'];
        for dataset in datasets:    
            print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
            datasetPath = "./input/"+dataset+"_prePCA";
            result = doExp(datasetPath,varianceRatio,numOfRounds,isLinearSVM=isLinearSVM);
            np.savetxt(resultSavedPath+"Epsilon_"+dataset+".output",result,delimiter=",",fmt='%1.3f');
