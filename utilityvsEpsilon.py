from pkg.svm import SVMModule;
from pkg.dimReduction import PCAModule;
from pkg.diffPrivDimReduction import DiffPrivPCAModule;
import numpy as np;
from sklearn.model_selection import ShuffleSplit;
import matplotlib.pyplot as plt;
import sys;
import os;

def drawF1Score(datasetTitle, data=None,path=None,figSavedPath=None):
    
    plt.clf();
    if path is not None:
        data = np.loadtxt(path,delimiter=",");
    x = data[:,0];
    data = data[:,[3,6,9]];
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
    
    x = data[:,0];
    data = data[:,[1,2,5,6,7,8]];
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
    xEpsilons = np.arange(0.1,1.0,0.1);
    cprResult = np.zeros((9,10)); 
    
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
        
        for k, targetEpsilon in np.ndenumerate(xEpsilons):
            #print pcaImpl.projMatrix[:,0];    
            print "epsilon: %.2f, delta: %f" % (targetEpsilon,delta);
                    
            isGaussianDist = True;
            dpGaussianPCAImpl.setEpsilonAndGamma(targetEpsilon,delta);
            dpGaussianPCAImpl.getDiffPrivPCs(isGaussianDist);
            
            isGaussianDist = False;
            dpWishartPCAImpl.setEpsilonAndGamma(targetEpsilon,delta);
            dpWishartPCAImpl.getDiffPrivPCs(isGaussianDist);
            '''
            We don't need to project the data multiple times.
            '''
            print "non noise PCA SVM training";
            if isLinearSVM:
                result = SVMModule.SVMClf.linearSVM(projTrainingData1,trainingLabel,projTestingData1,testingLabel);
            else:
                result = SVMModule.SVMClf.rbfSVM(projTrainingData1,trainingLabel,projTestingData1,testingLabel);
            
            cprResult[k][0] = cprResult[k][0]+targetEpsilon;
            cprResult[k][1] = cprResult[k][1]+result[0];
            cprResult[k][2] = cprResult[k][2]+result[1];
            cprResult[k][3] = cprResult[k][3]+result[2];
            
            projTrainingData2 = dpGaussianPCAImpl.transform(pureTrainingData,numOfDimension);
            projTestingData2 = dpGaussianPCAImpl.transform(pureTestingData,numOfDimension);
            print "Gaussian-DPPCA SVM training";
            
            if isLinearSVM:
                result = SVMModule.SVMClf.linearSVM(projTrainingData2,trainingLabel,projTestingData2,testingLabel);
            else:
                result = SVMModule.SVMClf.rbfSVM(projTrainingData2,trainingLabel,projTestingData2,testingLabel);
            cprResult[k][4] = cprResult[k][4]+result[0];
            cprResult[k][5] = cprResult[k][5]+result[1];
            cprResult[k][6] = cprResult[k][6]+result[2];
            
            projTrainingData3 = dpWishartPCAImpl.transform(pureTrainingData,numOfDimension);
            projTestingData3 = dpWishartPCAImpl.transform(pureTestingData,numOfDimension);
            #print projTestingData.shape;
            print "Wishart-DPPCA SVM training";
            if isLinearSVM:
                result = SVMModule.SVMClf.linearSVM(projTrainingData3,trainingLabel,projTestingData3,testingLabel);
            else:
                result = SVMModule.SVMClf.rbfSVM(projTrainingData3,trainingLabel,projTestingData3,testingLabel);
            cprResult[k][7] = cprResult[k][7]+result[0];
            cprResult[k][8] = cprResult[k][8]+result[1];
            cprResult[k][9] = cprResult[k][9]+result[2];
            
            print "===========================";
        """
        for i in range(0,len(cprResult)):
            print ','.join(['%.3f' % num for num in cprResult[i]]);
        """
    avgResult = cprResult/numOfRounds;
    
    for result in avgResult:
        print ','.join(['%.3f' % num for num in result]);
    
    return avgResult;    
if __name__ == "__main__":
    
    numOfRounds = 4;
    varianceRatio = 0.9;
    figSavedPath = "./log/";
    resultSavedPath = "./log/";
    if len(sys.argv) > 1:
        datasetPath = sys.argv[1];
        print "+++ using passed in arguments: %s" % (datasetPath);
        result = doExp(datasetPath,varianceRatio,numOfRounds,isLinearSVM=True);
        np.savetxt(resultSavedPath+"Epsilon_"+os.path.basename(datasetPath)+".output",result,delimiter=",");
    else:
        datasets = ['Amazon_3','face2','madelon','CNAE_2'];
        for dataset in datasets:    
            print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
            datasetPath = "./input/"+dataset+"_prePCA";
            result = doExp(datasetPath,varianceRatio,numOfRounds,isLinearSVM=True);
            np.savetxt(resultSavedPath+"Epsilon_"+dataset+".output",result,delimiter=",");
            #drawF1Score(dataset,data=result,figSavedPath=figSavedPath);
            #drawPrecisionRecall(dataset,data=result,figSavedPath=figSavedPath);