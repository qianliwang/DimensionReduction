from pkg.svm import SVMModule;
from pkg.dimReduction import PCAModule;
from pkg.diffPrivDimReduction import DiffPrivPCAModule;
import numpy as np;
from sklearn.model_selection import ShuffleSplit;
import matplotlib.pyplot as plt;

def drawF1Score(datasetTitle,data=None,path=None,figSavedPath=None):
    plt.clf();
    if path is not None:
        data = np.loadtxt(path,delimiter=",");
    xBound = len(data)+1;
    x = data[:,0];
    print x;
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

def doExp(datasetPath,numOfRounds,numOfDimensions,isLinearSVM=True):
    data = np.loadtxt(datasetPath,delimiter=",");
    rs = ShuffleSplit(n_splits=numOfRounds, test_size=.2, random_state=0);
    rs.get_n_splits(data);
    numOfFeature = data.shape[1]-1;
    
    if numOfDimensions > numOfFeature:
        numOfDimensions = numOfFeature;
        dataRange = 1;
    else:
        dataRange = numOfFeature/numOfDimensions;
    
    cprResult = np.zeros((numOfDimensions-1,4));
    
    for train_index, test_index in rs.split(data):
        
        trainingData = data[train_index];
        pureTrainingData = trainingData[:,1:];
        trainingLabel = trainingData[:,0];
        
        testingData = data[test_index];
        pureTestingData = testingData[:,1:];
        testingLabel = testingData[:,0];
        
        numOfTrainingSamples = trainingData.shape[0];
        
        pcaImpl = PCAModule.PCAImpl(pureTrainingData);
        pcaImpl.getPCs();
        
        epsilon = 0.5;
        delta = np.divide(1.0,numOfTrainingSamples);
        print "epsilon: %.2f, delta: %f" % (epsilon,delta);
        
        
        isGaussianDist = True;
        dpGaussianPCAImpl = DiffPrivPCAModule.DiffPrivPCAImpl(pureTrainingData);
        dpGaussianPCAImpl.setEpsilonAndGamma(epsilon,delta);
        
        isGaussianDist = False;
        dpWishartPCAImpl = DiffPrivPCAModule.DiffPrivPCAImpl(pureTrainingData);
        dpWishartPCAImpl.setEpsilonAndGamma(epsilon,delta);
        
        for k in range(1,numOfDimensions):
            
            #print pcaImpl.projMatrix[:,0];
            #result = SVMModule.SVMClf.rbfSVM(pureTrainingData,trainingLabel,pureTestingData,testingLabel);
            #print result;
            targetDimension = k*dataRange;
            cprResult[k-1][0] = cprResult[k-1][0]+targetDimension;
            projTrainingData1 = pcaImpl.transform(pureTrainingData,targetDimension);
            projTestingData1 = pcaImpl.transform(pureTestingData,targetDimension);
            #print projTrainingData.shape;
            if isLinearSVM:
                result = SVMModule.SVMClf.linearSVM(projTrainingData1,trainingLabel,projTestingData1,testingLabel);
            else:
                result = SVMModule.SVMClf.rbfSVM(projTrainingData1,trainingLabel,projTestingData1,testingLabel);
            
            cprResult[k-1][1] = cprResult[k-1][1]+result[2];
            
            isGaussianDist = True;
            dpGaussianPCAImpl.getDiffPrivPCs(isGaussianDist);
            projTrainingData2 = dpGaussianPCAImpl.transform(pureTrainingData,targetDimension);
            projTestingData2 = dpGaussianPCAImpl.transform(pureTestingData,targetDimension);
            #print projTestingData.shape;
            if isLinearSVM:
                result = SVMModule.SVMClf.linearSVM(projTrainingData2,trainingLabel,projTestingData2,testingLabel);
            else:
                result = SVMModule.SVMClf.rbfSVM(projTrainingData2,trainingLabel,projTestingData2,testingLabel);
            cprResult[k-1][2] = cprResult[k-1][2]+result[2];
            
            isGaussianDist = False;
            dpWishartPCAImpl.getDiffPrivPCs(isGaussianDist);
            projTrainingData3 = dpWishartPCAImpl.transform(pureTrainingData,targetDimension);
            projTestingData3 = dpWishartPCAImpl.transform(pureTestingData,targetDimension);
            #print projTestingData.shape;
            if isLinearSVM:
                result = SVMModule.SVMClf.linearSVM(projTrainingData3,trainingLabel,projTestingData3,testingLabel);
            else:
                result = SVMModule.SVMClf.rbfSVM(projTrainingData3,trainingLabel,projTestingData3,testingLabel);
            cprResult[k-1][3] = cprResult[k-1][3]+result[2];
            
            print "===========================";
            """
            for i in range(0,len(cprResult)):
                print "%f,%f,%f" % (cprResult[i][0],cprResult[i][1],cprResult[i][2]);
            """
    print "******************************";
    avgCprResult = cprResult/numOfRounds;
    for i in range(0,len(cprResult)):
        print "%.3f,%.3f,%.3f" % (avgCprResult[i][0],avgCprResult[i][1],avgCprResult[i][2]);
    return avgCprResult;
if __name__ == "__main__":
    
    #datasets = ['diabetes','german','ionosphere'];
    datasets = ['diabetes','CNAE_1',];
    numOfRounds = 10;
    figSavedPath = "./log/";
    numOfDimensions = 20;
    for dataset in datasets:
    
        print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
        
        datasetPath = "../distr_dp_pca/experiment/input/"+dataset+"_prePCA";
        result = doExp(datasetPath,numOfRounds,numOfDimensions,isLinearSVM=True);
        drawF1Score(dataset,data=result,figSavedPath=figSavedPath);    