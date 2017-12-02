from pkg.svm import SVMModule;
from pkg.dimReduction import PCAModule;
from pkg.diffPrivDimReduction import DiffPrivPCAModule;
import numpy as np;
from sklearn.model_selection import ShuffleSplit;
import matplotlib.pyplot as plt;
def drawF1Score(datasetTitle, data=None,path=None,figSavedPath=None):
    
    plt.clf();
    if path is not None:
        data = np.loadtxt(path,delimiter=",");
    x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9];
    data = data[:,[2,5,8]];
    minVector = np.amin(data,axis=0);
    yMin = min(minVector);
    maxVector = np.amax(data,axis=0);
    yMax = max(maxVector);
    
    yMin = (yMin-0.1) if (yMin-0.1)>0 else 0;
    yMax = (yMax+0.1) if (yMax+0.1)<1 else 1;
    #x = [10,40,70,100,130,160,190,220,250,280,310,340];
    y1Line,y2Line,y3Line = plt.plot(x, data[:,0], 'bo-', x, data[:,1], 'r^-',x, data[:,2], 'gs-');
    plt.legend([y1Line,y2Line,y3Line], ['PCA', 'Gaussian Noise','Wishart Noise'],loc=1);
    plt.axis([0.05,0.95,yMin,yMax]);
    #plt.axis([0,10,0.4,1.0]);
    plt.xlabel('Epsilon',fontsize=18);
    plt.ylabel('F1-Score',fontsize=18);
    plt.title(datasetTitle+' Dataset', fontsize=18);
    plt.xticks(x);
    if figSavedPath is None:
        plt.show();
    else:
        plt.savefig(figSavedPath+"epsilon_f1_"+datasetTitle+'.pdf', format='pdf', dpi=1000);
    
def drawPrecisionRecall(datasetTitle, data=None,path=None,figSavedPath=None):
    
    plt.clf();
    if path is not None:
        data = np.loadtxt(path,delimiter=",");
    
    x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9];
    data = data[:,[0,1,3,4,6,7]];
    minVector = np.amin(data,axis=0);
    yMin = min(minVector);
    maxVector = np.amax(data,axis=0);
    yMax = max(maxVector);
    
    yMin = (yMin-0.2) if (yMin-0.2)>0 else 0;
    yMax = (yMax+0.2) if (yMax+0.2)<1 else 1;
    #x = [10,40,70,100,130,160,190,220,250,280,310,340];
    y1Line,y2Line,y3Line = plt.plot(x, data[:,0], 'bo-', x, data[:,2], 'r^-',x, data[:,4], 'gs-');
    y4Line,y5Line,y6Line = plt.plot(x, data[:,1], 'bo--', x, data[:,3], 'r^--',x, data[:,5], 'gs--');
    plt.legend([y1Line,y2Line,y3Line,y4Line,y5Line,y6Line], ['Precision-PCA', 'Precision-Gaussian Noise','Precision-Wishart Noise','Recall-PCA', 'Recall-Gaussian Noise','Recall-Wishart Noise'],loc=4);
    if datasetTitle == "german":
        plt.axis([0.05,0.95,0,0.75]);
    elif datasetTitle == "Ionosphere":
        plt.axis([0.05,0.95,0.4,1]);
    else:
        plt.axis([0.05,0.95,0,1]);
    #plt.axis([0,10,0.4,1.0]);
    plt.xlabel('Epsilon',fontsize=18);
    plt.ylabel('Precision & Recall',fontsize=18);
    plt.title(datasetTitle+' Dataset', fontsize=18);
    plt.xticks(x);
    if figSavedPath is None:
        plt.show();
    else:
        plt.savefig(figSavedPath+"epsilon_pr_"+datasetTitle+'.pdf', format='pdf', dpi=1000);

def doExp(datasetPath,varianceRatio,numOfRounds,isLinearSVM=True):
    
    data = np.loadtxt(datasetPath,delimiter=",");
    rs = ShuffleSplit(n_splits=numOfRounds, test_size=.2, random_state=0);
    rs.get_n_splits(data);
    
    cprResult = np.zeros((9,9)); 
    
    for train_index, test_index in rs.split(data):
        
        trainingData = data[train_index];
        pureTrainingData = trainingData[:,1:];
        trainingLabel = trainingData[:,0];
        
        numOfTrainingSamples = trainingData.shape[0];
        numOfFeature = trainingData.shape[1]-1;
        
        testingData = data[test_index];
        pureTestingData = testingData[:,1:];
        testingLabel = testingData[:,0];
        
        pcaImpl = PCAModule.PCAImpl(pureTrainingData);
        pcaImpl.getPCs();
        numOfDimension = pcaImpl.getNumOfPCwithKPercentVariance(varianceRatio);
        print "%d/%d dimensions captures %.2f variance." % (numOfDimension,numOfFeature,varianceRatio);
        
        dpGaussianPCAImpl = DiffPrivPCAModule.DiffPrivPCAImpl(pureTrainingData);
        dpWishartPCAImpl = DiffPrivPCAModule.DiffPrivPCAImpl(pureTrainingData);
        
        delta = np.divide(1.0,numOfTrainingSamples);
        projTrainingData1 = pcaImpl.transform(pureTrainingData,numOfDimension);
        projTestingData1 = pcaImpl.transform(pureTestingData,numOfDimension);
        #print projTrainingData.shape;
        
        
        for k in range(1,10):
            #print pcaImpl.projMatrix[:,0];    
            epsilon = 0.1*k;
            print "epsilon: %.2f, delta: %f" % (epsilon,delta);
                    
            isGaussianDist = True;
            dpGaussianPCAImpl.setEpsilonAndGamma(epsilon,delta);
            dpGaussianPCAImpl.getDiffPrivPCs(isGaussianDist);
            
            isGaussianDist = False;
            dpWishartPCAImpl.setEpsilonAndGamma(epsilon,delta);
            dpWishartPCAImpl.getDiffPrivPCs(isGaussianDist);
            '''
            We don't need to project the data multiple times.
            '''
            print "-PCA SVM training";
            if isLinearSVM:
                result = SVMModule.SVMClf.linearSVM(projTrainingData1,trainingLabel,projTestingData1,testingLabel);
            else:
                result = SVMModule.SVMClf.rbfSVM(projTrainingData1,trainingLabel,projTestingData1,testingLabel);
            
            cprResult[k-1][0] = cprResult[k-1][0]+result[0];
            cprResult[k-1][1] = cprResult[k-1][1]+result[1];
            cprResult[k-1][2] = cprResult[k-1][2]+result[2];
            
            projTrainingData2 = dpGaussianPCAImpl.transform(pureTrainingData,numOfDimension);
            projTestingData2 = dpGaussianPCAImpl.transform(pureTestingData,numOfDimension);
            print "-Gaussian-DPPCA SVM training";
            
            if isLinearSVM:
                result = SVMModule.SVMClf.linearSVM(projTrainingData2,trainingLabel,projTestingData2,testingLabel);
            else:
                result = SVMModule.SVMClf.rbfSVM(projTrainingData2,trainingLabel,projTestingData2,testingLabel);
            cprResult[k-1][3] = cprResult[k-1][3]+result[0];
            cprResult[k-1][4] = cprResult[k-1][4]+result[1];
            cprResult[k-1][5] = cprResult[k-1][5]+result[2];
            
            projTrainingData3 = dpWishartPCAImpl.transform(pureTrainingData,numOfDimension);
            projTestingData3 = dpWishartPCAImpl.transform(pureTestingData,numOfDimension);
            #print projTestingData.shape;
            if isLinearSVM:
                result = SVMModule.SVMClf.linearSVM(projTrainingData3,trainingLabel,projTestingData3,testingLabel);
            else:
                result = SVMModule.SVMClf.rbfSVM(projTrainingData3,trainingLabel,projTestingData3,testingLabel);
            cprResult[k-1][6] = cprResult[k-1][6]+result[0];
            cprResult[k-1][7] = cprResult[k-1][7]+result[1];
            cprResult[k-1][8] = cprResult[k-1][8]+result[2];
            
            print "===========================";
        """
        for i in range(0,len(cprResult)):
            print ','.join(['%.3f' % num for num in cprResult[i]]);
        """
    avgResult = cprResult/numOfRounds;
    
    for i in range(0,len(avgResult)):
        print ','.join(['%.3f' % num for num in avgResult[i]]);
    
    return avgResult;    
if __name__ == "__main__":
    
    datasets = ['diabetes','german','ionosphere'];
    numOfRounds = 10;
    varianceRatio = 0.9;
    figSavedPath = "./log/";
    
    for dataset in datasets:    
        print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
        datasetPath = "../distr_dp_pca/experiment/input/"+dataset+"_prePCA";
        result = doExp(datasetPath,varianceRatio,numOfRounds,isLinearSVM=True);
        drawF1Score(dataset,data=result,figSavedPath=figSavedPath);
        drawPrecisionRecall(dataset,data=result,figSavedPath=figSavedPath);