from pkg.svm import SVMModule;
from pkg.dimReduction import PCAModule;
from pkg.diffPrivDimReduction import DiffPrivPCAModule;
import numpy as np;
from sklearn.model_selection import ShuffleSplit;
import matplotlib.pyplot as plt;
import sys;
import os;
from multiprocessing import Pool;
from sklearn.preprocessing import StandardScaler;


def drawF1Score(datasetTitle,data=None,path=None,figSavedPath=None):
    plt.clf();
    if path is not None:
        data = np.loadtxt(path,delimiter=",");
    xBound = len(data)+1;
    x = data[:,0];
    
    minVector = np.amin(data[:,1:],axis=0);
    yMin = min(minVector);
    maxVector = np.amax(data[:,1:],axis=0);
    yMax = max(maxVector);
    
    yMin = (yMin-0.1) if (yMin-0.1)>0 else 0;
    yMax = (yMax+0.1) if (yMax+0.1)<1 else 1;
    #x = [10,40,70,100,130,160,190,220,250,280,310,340];
    y1Line,y2Line,y3Line = plt.plot(x, data[:,1], 'bo-', x, data[:,2], 'r^-',x, data[:,3], 'gs-');
    
    plt.legend([y1Line,y2Line,y3Line], ['PCA', 'Gaussian Noise','Wishart Noise'],loc=4);
    plt.axis([0,xBound,yMin,yMax]);
    #plt.axis([0,10,0.4,1.0]);
    plt.xlabel('Number of Principal Components',fontsize=18);
    plt.ylabel('F1-Score',fontsize=18);
    plt.title(datasetTitle+' Dataset', fontsize=18);
    plt.xticks(x);
    if figSavedPath is None:
        plt.show();
    else:
        plt.savefig(figSavedPath+"numOfPC_"+datasetTitle+'.pdf', format='pdf', dpi=1000);

def singleExp(xDimensions,trainingData,testingData,largestReducedFeature,isLinearSVM):
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

    cprResult = np.zeros((len(xDimensions),4));
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
        
        """
        for result in cprResult:
            print "%f,%f,%f" % (result[0],result[1],result[2]);
        """
    return cprResult;

def doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfDimensions,isLinearSVM=True):
    data = np.loadtxt(datasetPath,delimiter=",");
    globalPCA = PCAModule.PCAImpl(data[:,1:]);
    
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
    rs = ShuffleSplit(n_splits=numOfRounds, test_size=.2, random_state=0);
    rs.get_n_splits(data);
    #p = Pool(numOfRounds);
    
    for train_index, test_index in rs.split(data):    
        trainingData = data[train_index];
        testingData = data[test_index];
        
        #tmpResult = p.apply_async(singleExp, (xDimensions,trainingData,testingData,largestReducedFeature,isLinearSVM));

        #cprResult += tmpResult.get();
        tmpResult = singleExp(xDimensions, trainingData, testingData, largestReducedFeature, isLinearSVM);
        if cprResult is None:
            cprResult = tmpResult;
        else:
            cprResult = np.concatenate((cprResult,tmpResult),axis=0);
    #avgCprResult = cprResult/numOfRounds;
    avgCprResult = cprResult;
    for result in avgCprResult:
        print "%d,%.3f,%.3f,%.3f" % (result[0],result[1],result[2],result[3]);
    #p.close();
    #p.join();
    return avgCprResult;

if __name__ == "__main__":
    #datasets = ['diabetes','german','ionosphere'];
    numOfRounds = 4;
    figSavedPath = "./log/";
    resultSavedPath = "./log/";
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
        datasets = ['diabetes','CNAE_2','CNAE_5','CNAE_7','face2','Amazon_3','madelon'];
        #datasets = ['diabetes','Amazon_2','Australian','german','ionosphere'];
        for dataset in datasets:
            print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
            datasetPath = "./input/"+dataset+"_prePCA";
            result = doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfDimensions,isLinearSVM=isLinearSVM);
            np.savetxt(resultSavedPath+"numPC_"+dataset+".output",result,delimiter=",",fmt='%1.3f');
            #drawF1Score(dataset,data=result,figSavedPath=figSavedPath);    