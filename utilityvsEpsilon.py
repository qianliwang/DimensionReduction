from pkg.svm import SVMModule;
from pkg.dimReduction import PCAModule;
from pkg.diffPrivDimReduction import DiffPrivPCAModule;
import numpy as np;
from numpy import linalg as LA;
from sklearn.model_selection import ShuffleSplit;
import matplotlib.pyplot as plt;
import sys;
import os;
from multiprocessing import Pool;
from sklearn.preprocessing import StandardScaler;
from pkg.global_functions import globalFunction as gf;

def drawF1Score(datasetTitle, data=None,path=None,figSavedPath=None):
    
    plt.clf();
    if path is not None:
        data = np.loadtxt(path,delimiter=",");
    x = data[:9,0];
    pcaF1 = data[np.arange(0,90,9),3];
    pcaF1Mean = np.full((9,),np.mean(pcaF1));
    pcaF1Std = np.full((9,),0);
    pcaF1Line = plt.errorbar(x, pcaF1Mean/2, yerr=pcaF1Std, fmt='b-',capsize=4);
    gF1 = [];
    wF1=[];
    for i in range(0,9):
       gIndices = np.arange(i, 90, 9);
       gF1.append(data[gIndices,6]);
       wF1.append(data[gIndices,9]);
    #print np.asarray(gF1);
    gF1Mean,gF1Std = gf.calcMeanandStd(np.asarray(gF1).T)
    gF1Line = plt.errorbar(x, gF1Mean, yerr=gF1Std, fmt='g-',capsize=4);
     
    wF1Mean,wF1Std = gf.calcMeanandStd(np.asarray(wF1).T)
    wF1Line = plt.errorbar(x, wF1Mean, yerr=wF1Std, fmt='r-',capsize=4);

    """
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
    """
    plt.axis([0.05,0.95,0,1.1]);
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
    x = data[:9, 0];
    pcaPrecision = data[np.arange(0, 90, 9), 1];
    pcaPrecMean = np.full((9,), np.mean(pcaPrecision));
    pcaPrecStd = np.full((9,), 0);
    pcaPrecLine = plt.errorbar(x, pcaPrecMean / 2, yerr=pcaPrecStd, fmt='b-',capsize=4);
    gPrec = [];
    wPrec = [];
    for i in range(0, 9):
        gIndices = np.arange(i, 90, 9);
        gPrec.append(data[gIndices, 4]);
        wPrec.append(data[gIndices, 7]);
    # print np.asarray(gF1);
    gPrecMean, gPrecStd = gf.calcMeanandStd(np.asarray(gPrec).T)
    gPrecLine = plt.errorbar(x, gPrecMean, yerr=gPrecStd, fmt='g-',capsize=4);
        
    wPrecMean, wPrecStd = gf.calcMeanandStd(np.asarray(wPrec).T)
    wPrecLine = plt.errorbar(x, wPrecMean, yerr=wPrecStd, fmt='r-',capsize=4);


    pcaRecall = data[np.arange(0, 90, 9), 2];
    pcaRecMean = np.full((9,), np.mean(pcaRecall));
    pcaRecStd = np.full((9,), 0);
    pcaRecLine = plt.errorbar(x, pcaRecMean / 2, yerr=pcaRecStd, fmt='b--',capsize=4);
    gRec = [];
    wRec = [];
    for i in range(0, 9):
        gIndices = np.arange(i, 90, 9);
        gRec.append(data[gIndices, 5]);
        wRec.append(data[gIndices, 8]);
    # print np.asarray(gF1);
    gRecMean, gRecStd = gf.calcMeanandStd(np.asarray(gRec).T)
    gRecLine = plt.errorbar(x, gRecMean, yerr=gRecStd, fmt='g--',capsize=4);
       
    wRecMean, wRecStd = gf.calcMeanandStd(np.asarray(wRec).T)
    wRecLine = plt.errorbar(x, wRecMean, yerr=wRecStd, fmt='r--',capsize=4);
    """
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
    """
    plt.axis([0,1,0,1.1]);
    plt.xlabel('Epsilon',fontsize=18);
    plt.ylabel('Precision & Recall',fontsize=18);
    plt.title(datasetTitle+' Dataset', fontsize=18);
    plt.xticks(x);
    if figSavedPath is None:
        plt.show();
    else:
        plt.savefig(figSavedPath+"epsilon_pr_"+datasetTitle+'.pdf', format='pdf', dpi=1000);

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
    
def doExp(datasetPath,varianceRatio,numOfRounds,isLinearSVM=True):
    
    data = np.loadtxt(datasetPath,delimiter=",");
    rs = ShuffleSplit(n_splits=numOfRounds, test_size=.2, random_state=0);
    rs.get_n_splits(data);
    globalPCA = PCAModule.PCAImpl(data[:,1:]);
    numOfFeature = data.shape[1]-1;
    largestReducedFeature = globalPCA.getNumOfPCwithKPercentVariance(varianceRatio);
    print "%d/%d dimensions captures %.2f variance." % (largestReducedFeature,numOfFeature,varianceRatio);
    
    xEpsilons = np.arange(0.1,1.0,0.1);
    cprResult = None;
    #cprResult = np.zeros((len(xEpsilons),10));
    #p = Pool(numOfRounds);
    
    #normalizedData = normByRow(data[:,1:]);
    #normalizedData = np.concatenate((data[:,[0,]],normalizedData),axis=1);

    for train_index, test_index in rs.split(data):
        trainingData = data[train_index];
        testingData = data[test_index];
        #tmpResult = p.apply_async(singleExp, (xEpsilons,trainingData,testingData,largestReducedFeature,isLinearSVM));
        #cprResult += tmpResult.get();
        tmpResult = singleExp(xEpsilons,trainingData,testingData,largestReducedFeature,isLinearSVM);
        if cprResult is None:
            cprResult = tmpResult;
        else:
            cprResult = np.concatenate((cprResult,tmpResult),axis=0);
    # To record the mean and standard deviation.
    avgResult = cprResult;
    #avgResult = cprResult/numOfRounds;
    #p.close();
    #p.join();
    for result in avgResult:
        print ','.join(['%.3f' % num for num in result]);
    
    return avgResult;    
if __name__ == "__main__":
    
    numOfRounds = 10;
    varianceRatio = 0.9;
    figSavedPath = "./log/";
    resultSavedPath = "./log/";
    isLinearSVM = True ;
    if len(sys.argv) > 1:
        datasetPath = sys.argv[1];
        print "+++ using passed in arguments: %s" % (datasetPath);
        result = doExp(datasetPath,varianceRatio,numOfRounds,isLinearSVM=isLinearSVM);
        np.savetxt(resultSavedPath+"Epsilon_"+os.path.basename(datasetPath)+".output",result,delimiter=",",fmt='%1.3f');
    else:
        #datasets = ['diabetes','CNAE_2','ionosphere','CNAE_5','CNAE_7','face2','Amazon_3','madelon'];
        datasets = ['p53_3000','CNAE_2','B11_10','Amazon_3','ionosphere','CNAE_5','CNAE_7','face2','Amazon_3','madelon'];
        for dataset in datasets:
            print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
            datasetPath = "./input/"+dataset+"_prePCA";
            #result = doExp(datasetPath,varianceRatio,numOfRounds,isLinearSVM=isLinearSVM);
            #np.savetxt(resultSavedPath+"Epsilon_"+dataset+".output",result,delimiter=",",fmt='%1.3f');
            drawF1Score(dataset,data=None,path = resultSavedPath+"Epsilon_"+dataset+".output",figSavedPath=None);
            #drawPrecisionRecall(dataset,data=None,path =resultSavedPath+"Epsilon_"+dataset+".output", figSavedPath=None);