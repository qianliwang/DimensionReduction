from sklearn.model_selection import StratifiedShuffleSplit;
from sklearn.preprocessing import StandardScaler;
from sklearn import preprocessing;
from sklearn.model_selection import ShuffleSplit;

from pkg import SVMModule;
from pkg.DimReduction import PCAImpl;
from pkg.DPDimReduction import DiffPrivPCAImpl;
from pkg.DPDimReduction import DiffPrivImpl;
from pkg.global_functions import globalFunction as gf;

import numpy as np;
import sys;
import os;

def drawF1Score(datasetTitle,data=None,path=None,n_trails=1,figSavedPath=None):
    import matplotlib.pyplot as plt;
    from matplotlib.ticker import MultipleLocator;

    plt.clf();
    if path is not None:
        data = np.loadtxt(path,delimiter=",");

    """
    #x = [10,40,70,100,130,160,190,220,250,280,310,340];
    y1Line,y2Line,y3Line = plt.plot(x, data[:,1], 'bo-', x, data[:,2], 'r^-',x, data[:,3], 'gs-');
    
    plt.legend([y1Line,y2Line,y3Line], ['PCA', 'Gaussian Noise','Wishart Noise'],loc=4);
    """
    minVector = np.amin(data[:, 1:], axis=0);
    yMin = min(minVector);
    maxVector = np.amax(data[:, 1:], axis=0);
    yMax = max(maxVector);

    yMin = (yMin - 0.05) if (yMin - 0.05) > 0 else 0;
    yMax = (yMax + 0.05) if (yMax + 0.05) < 1 else 1.05;
    n_dim = data.shape[0];

    if n_trails is not 1:
        n_dim = int(data.shape[0] / n_trails);
        data = data.reshape(n_trails,-1,data.shape[1]);
        data_mean,data_std = gf.calcMeanandStd(data);
    else:
        data_mean = data;
        data_std = np.zeros(data.shape);
    print "Number of points on x-axis: %d" % n_dim;
    #print(data_mean);
    #print(data_std);
    x = data_mean[:, 0];

    pcaF1Mean = data_mean[:,3];
    pcaF1Std = data_std[:,3];
    largestXVal = x[-1];

    pcaF1ErrorLine = plt.errorbar(x, pcaF1Mean, yerr=pcaF1Std, fmt='b',capsize=4);
    pcaF1Line, = plt.plot(x, pcaF1Mean, 'b-')

    gF1Mean = data_mean[:, 7];
    gF1Std = data_std[:, 7];
    gF1ErrorLine = plt.errorbar(x, gF1Mean, yerr=gF1Std, fmt='m',capsize=4);
    gF1Line, = plt.plot(x, gF1Mean, 'm-')

    gF1Mean = data_mean[:, 11];
    gF1Std = data_std[:, 11];
    #wF1ErrorLine = plt.errorbar(x, wF1Mean, yerr=wF1Std, fmt='g',capsize=4);
    #wF1Line, = plt.plot(x, wF1Mean , 'g-')
    plt.axis([0,x[-1]+1,yMin,yMax]);
    #plt.axis([0,10,0.4,1.0]);
    #plt.legend([gF1Line, wF1Line,pcaF1Line], ['Gaussian Noise', 'Wishart Noise', 'PCA'], loc=4);
    plt.legend([gF1Line, pcaF1Line], ['DPDPCA', 'PCA'], loc=4);
    plt.xlabel('Number of Principal Components',fontsize=18);
    plt.ylabel('F1-Score',fontsize=18);
    plt.title(datasetTitle, fontsize=18);
    plt.xticks(x);
    ax = plt.gca();
    if largestXVal>50:
        majorLocator = MultipleLocator(8);
    else:
        majorLocator = MultipleLocator(2);
    ax.xaxis.set_major_locator(majorLocator);
    if figSavedPath is None:
        plt.show();
    else:
        plt.savefig(figSavedPath+"numOfPC_"+datasetTitle+'.pdf', format='pdf', dpi=1000);

def singleExp(xDimensions,trainingData,testingData,largestReducedFeature,epsilon,isLinearSVM):
    pureTrainingData = trainingData[:, 1:];
    trainingLabel = trainingData[:, 0];

    pureTestingData = testingData[:, 1:];
    testingLabel = testingData[:, 0];

    scaler = StandardScaler();
    # print pureTrainingData[0];
    #scaler.fit(pureTrainingData);
    pureTrainingData = scaler.fit_transform(pureTrainingData);
    # print pureTrainingData[0];

    # print pureTestingData[0];
    pureTestingData = scaler.transform(pureTestingData);
    # print pureTestingData[0];

    cprResult = [];
    pcaImpl = PCAImpl(pureTrainingData);

    pcaImpl.getPCs(largestReducedFeature);
    numOfTrainingSamples = trainingData.shape[0];

    delta = np.divide(1.0, numOfTrainingSamples);
    print "epsilon: %.2f, delta: %f" % (epsilon, delta);
    
    isGaussianDist = True;
    dpGaussianPCAImpl = DiffPrivPCAImpl(pureTrainingData);
    dpGaussianPCAImpl.setEpsilonAndGamma(epsilon,delta);
    dpGaussianPCAImpl.getDiffPrivPCs(isGaussianDist,largestReducedFeature);
    
    isGaussianDist = False;
    dpWishartPCAImpl = DiffPrivPCAImpl(pureTrainingData);
    dpWishartPCAImpl.setEpsilonAndGamma(epsilon,delta);
    dpWishartPCAImpl.getDiffPrivPCs(isGaussianDist,largestReducedFeature);
    
    for k, targetDimension in np.ndenumerate(xDimensions):    
        #print pcaImpl.projMatrix[:,0];
        #result = SVMModule.SVMClf.rbfSVM(pureTrainingData,trainingLabel,pureTestingData,testingLabel);
        #print k;
        cprResult.append(targetDimension);
        projTrainingData1 = pcaImpl.transform(pureTrainingData,targetDimension);
        projTestingData1 = pcaImpl.transform(pureTestingData,targetDimension);
        print "Non-noise PCA %d" % targetDimension;
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData1,trainingLabel,projTestingData1,testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData1,trainingLabel,projTestingData1,testingLabel);
        
        cprResult.extend(result);
        
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
        cprResult.extend(result);

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
        cprResult.extend(result);

        """
        for result in cprResult:
            print "%f,%f,%f" % (result[0],result[1],result[2]);
        """
    #print(cprResult);
    cprResult = np.asarray(cprResult);
    return cprResult.reshape((len(xDimensions),-1));

def doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfDimensions,logPath,isLinearSVM=True):
    if os.path.basename(datasetPath).endswith('npy'):
        data = np.load(datasetPath);
    else:
        data = np.loadtxt(datasetPath, delimiter=",");
    scaler = StandardScaler();
    data_std = scaler.fit_transform(data[:, 1:]);
    globalPCA = PCAImpl(data_std);

    numOfFeature = data.shape[1] - 1;
    largestReducedFeature = globalPCA.getNumOfPCwithKPercentVariance(varianceRatio);
    print "%d/%d dimensions captures %.2f variance." % (largestReducedFeature, numOfFeature, varianceRatio);
    xDimensions = None;
    if numOfDimensions > numOfFeature:
        xDimensions = np.arange(1, numOfFeature);
        largestReducedFeature = numOfFeature;
    else:
        xDimensions = np.arange(1, largestReducedFeature, max(largestReducedFeature / numOfDimensions, 1));

    cprResult = [];
    rs = StratifiedShuffleSplit(n_splits=numOfRounds, test_size=.2, random_state=0);
    rs.get_n_splits(data[:, 1:], data[:, 0]);

    for train_index, test_index in rs.split(data[:, 1:], data[:, 0]):
        trainingData = data[train_index];
        testingData = data[test_index];

        tmpResult = singleExp(xDimensions, trainingData, testingData, largestReducedFeature, epsilon,isLinearSVM);
        with open(logPath, "a") as f:
            np.savetxt(f,tmpResult,delimiter=",",fmt='%1.3f');
        cprResult.append(tmpResult);

    cprResult = np.vstack(cprResult);
    for result in cprResult:
        print ','.join(['%.3f' % num for num in result]);

    return cprResult;

if __name__ == "__main__":
    #datasets = ['diabetes','german','ionosphere'];
    numOfRounds = 2;
    figSavedPath = "./fig/";
    logSavedPath = "./log/";
    resultSavedPath = logSavedPath+"/firstRevision/";
    numOfDimensions = 30;
    epsilon = 0.3;
    varianceRatio = 0.9;
    isLinearSVM = True;
    if len(sys.argv) > 1:
        datasetPath = sys.argv[1];
        print "+++ using passed in arguments: %s" % (datasetPath);
        result = doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfDimensions,isLinearSVM=isLinearSVM);
        np.savetxt(resultSavedPath+"numPC_"+os.path.basename(datasetPath)+".output",result,delimiter=",",fmt='%1.3f');
    else:
        #datasets = ['diabetes','CNAE_2','CNAE_5','CNAE_7','face2','Amazon_3','madelon'];
        datasets = ['CNAE','Australian','YaleB','p53 Mutant','Amazon_2','german','ionosphere'];
        for dataset in datasets:
            print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
            datasetPath = "./input/"+dataset+"_prePCA";
            #result = doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfDimensions,logPath=logSavedPath+'numPC_'+dataset+".out",isLinearSVM=isLinearSVM);
            #np.savetxt(resultSavedPath+"numPC_"+dataset+".output",result,delimiter=",",fmt='%1.3f');
            drawF1Score(dataset,data=None,path = resultSavedPath+"numPC_"+dataset+".output",n_trails=10,figSavedPath=None);