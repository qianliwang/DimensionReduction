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
from pkg.global_functions import globalFunction as gf;
from matplotlib.ticker import MultipleLocator;

def drawF1Score(datasetTitle,data=None,path=None,figSavedPath=None):
    plt.clf();
    if path is not None:
        data = np.loadtxt(path,delimiter=",");
    numOfDim = data.shape[0]/10;
    x = data[:numOfDim, 0];
    print "Number of points on x-axis: %d" % numOfDim;
    """
    minVector = np.amin(data[:,1:],axis=0);
    yMin = min(minVector);
    maxVector = np.amax(data[:,1:],axis=0);
    yMax = max(maxVector);
    
    yMin = (yMin-0.1) if (yMin-0.1)>0 else 0;
    yMax = (yMax+0.1) if (yMax+0.1)<1 else 1;
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
    pcaF1 = [];
    gF1 = [];
    wF1 = [];
    for i in range(0, numOfDim):
        pcaIndices = np.arange(i, data.shape[0], numOfDim);
        pcaF1.append(data[pcaIndices, 1]);
        gF1.append(data[pcaIndices, 2]);
        wF1.append(data[pcaIndices, 3]);
    # print np.asarray(gF1);
    displayDim = 24;
    x = x[:displayDim];
    largestXVal = x[-1];
    pcaF1Mean, pcaF1Std = gf.calcMeanandStd(np.asarray(pcaF1).T);
    pcaF1Mean = pcaF1Mean[:displayDim];
    pcaF1Std = pcaF1Std[:displayDim];
    pcaF1ErrorLine = plt.errorbar(x, pcaF1Mean, yerr=pcaF1Std, fmt='b',capsize=4);
    pcaF1Line, = plt.plot(x, pcaF1Mean, 'b-')
    gF1Mean, gF1Std = gf.calcMeanandStd(np.asarray(gF1).T);
    gF1Mean = gF1Mean[:displayDim];
    gF1Std = gF1Std[:displayDim];
    gF1ErrorLine = plt.errorbar(x, gF1Mean, yerr=gF1Std, fmt='r',capsize=4);
    gF1Line, = plt.plot(x, gF1Mean, 'r-')
    wF1Mean, wF1Std = gf.calcMeanandStd(np.asarray(wF1).T);
    wF1Mean = wF1Mean[:displayDim];
    wF1Std = wF1Std[:displayDim];
    wF1ErrorLine = plt.errorbar(x, wF1Mean, yerr=wF1Std, fmt='g',capsize=4);
    wF1Line, = plt.plot(x, wF1Mean , 'g-')
    plt.axis([0,x[-1]+1,yMin,yMax]);
    #plt.axis([0,10,0.4,1.0]);
    plt.legend([pcaF1Line, gF1Line, wF1Line], ['PCA', 'Gaussian Noise', 'Wishart Noise'], loc=4);
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
    figSavedPath = "./fig/";
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
        #datasets = ['diabetes','CNAE_2','CNAE_5','CNAE_7','face2','Amazon_3','madelon'];
        datasets = ['CNAE','YaleB','p53 Mutant','Amazon_2','Australian','german','ionosphere'];
        for dataset in datasets:
            print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
            datasetPath = "./input/"+dataset+"_prePCA";
            #result = doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfDimensions,isLinearSVM=isLinearSVM);
            #np.savetxt(resultSavedPath+"numPC_"+dataset+".output",result,delimiter=",",fmt='%1.3f');
            drawF1Score(dataset,data=None,path = resultSavedPath+"numPC_"+dataset+".output",figSavedPath=figSavedPath);