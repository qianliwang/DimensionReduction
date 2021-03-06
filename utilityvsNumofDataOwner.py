from pkg.svm import SVMModule;
from pkg.dimReduction import PCAModule;
#from pkg.DPDPCA.DataOwner import DataOwnerImpl;
import numpy as np;
from numpy import linalg as LA;
from pkg.diffPrivDimReduction import invwishart;
from numpy.linalg import norm;
from sklearn.model_selection import ShuffleSplit;
from pkg.diffPrivDimReduction.DPModule import DiffPrivImpl;
import matplotlib.pyplot as plt;
import sys;
import os;
from multiprocessing import Pool;
import scipy.sparse as sparse;
from sklearn.preprocessing import StandardScaler;
from pkg.global_functions import globalFunction as gf;
from matplotlib.ticker import MultipleLocator;
from time import time;

def drawF1Score(datasetTitle, data=None, path=None, figSavedPath=None):
    plt.clf();
    if path is not None:
        data = np.loadtxt(path, delimiter=",");
    numOfDim = data.shape[0] / 9;
    x = data[:numOfDim, 0];
    xBound = x[-1] + 1;
    """
    minVector = np.amin(data[:,1:],axis=0);
    yMin = min(minVector);
    maxVector = np.amax(data[:,1:],axis=0);
    yMax = max(maxVector);

    yMin = (yMin-0.1) if (yMin-0.1)>0 else 0;
    yMax = (yMax+0.1) if (yMax+0.1)<1 else 1;
    #x = [10,40,70,100,130,160,190,220,250,280,310,340];
    y1Line,y2Line,y3Line = plt.plot(x, data[:,1], 'bo-', x, data[:,2], 'r^-',x, data[:,3], 'gs-');
    if datasetTitle is 'Ionosphere':
        plt.legend([y1Line,y2Line,y3Line], ['PCA','DPDPCA','PrivateLocalPCA'],loc=4);
    else:
        plt.legend([y1Line,y2Line,y3Line], ['PCA','DPDPCA','PrivateLocalPCA'],loc=2);
    """
    pcaF1 = [];
    dpdpcaF1 = [];
    privateF1 = [];
    for i in range(0, numOfDim):
        pcaIndices = np.arange(i, data.shape[0], numOfDim);
        pcaF1.append(data[pcaIndices, 1]);
        dpdpcaF1.append(data[pcaIndices, 2]);
        privateF1.append(data[pcaIndices, 3]);
    # print np.asarray(gF1);
    pcaF1Mean, pcaF1Std = gf.calcMeanandStd(np.asarray(pcaF1).T);
    pcaF1ErrorLine = plt.errorbar(x, pcaF1Mean, yerr=pcaF1Std, fmt='b', capsize=4);
    pcaF1Line, = plt.plot(x,pcaF1Mean,'b-');
    dpdpcaF1Mean, dpdpcaF1Std = gf.calcMeanandStd(np.asarray(dpdpcaF1).T);
    dpdpcaF1ErrorLine = plt.errorbar(x, dpdpcaF1Mean, yerr=dpdpcaF1Std, fmt='m', capsize=4);
    dpdpcaF1Line, = plt.plot(x,dpdpcaF1Mean,'m-');
    privateF1Mean, privateF1Std = gf.calcMeanandStd(np.asarray(privateF1).T);
    privateF1ErrorLine = plt.errorbar(x, privateF1Mean, yerr=privateF1Std, fmt='c', capsize=4);
    privateF1Line, = plt.plot(x,privateF1Mean,'c-');
    plt.axis([0, xBound, -0.05, 1.05]);
    if 'p53' in datasetTitle:
        plt.legend([pcaF1Line, dpdpcaF1Line, privateF1Line], ['PCA', 'DPDPCA', 'PrivateLocalPCA'], loc=2, fontsize='small');
    else:
        plt.legend([pcaF1Line, dpdpcaF1Line, privateF1Line], ['PCA', 'DPDPCA', 'PrivateLocalPCA'], loc=4, fontsize='small');
    # plt.axis([0,10,0.4,1.0]);
    plt.xlabel('Number of Principal Components', fontsize=18);
    plt.ylabel('F1-Score', fontsize=18);
    plt.title(datasetTitle, fontsize=18);
    plt.xticks(x);
    ax = plt.gca();
    if x[-1] > 100:
        majorLocator = MultipleLocator(8);
    else:
        majorLocator = MultipleLocator(2);
    ax.xaxis.set_major_locator(majorLocator);
    if figSavedPath is None:
        plt.show();
    else:
        plt.savefig(figSavedPath + "dataOwner_" + datasetTitle + '.pdf', format='pdf', dpi=1000);

def getApproxEigval(covMatrix,r1):
        temp1 = np.dot(covMatrix,r1);
        v1 = np.dot(r1.T,temp1);
        v2 = np.dot(r1.T,r1);
        eigVal = np.divide(v1,v2);
        return eigVal;

def genEigenvectors_power(covMatrix,topK):
    '''
    Compute the eigenvector with power iteration method, multiplying covariance with random vector, 
    converge threshold is setted through epsilon.
    '''
#    r0 = np.random.rand(covMatrix.shape[0],1);
    epsilon = 0.01;
    eigValues = np.zeros(topK);
    eigVectors = None;
    k=0;
    vecLength = covMatrix.shape[0];
    bound = min(1000,max(100,vecLength));
    while k<topK:
        r0 = np.random.rand(vecLength,1);
        count=0;
        while count<bound:
            r1 = np.dot(covMatrix, r0);
            # Get the second norm of r1;
            scale = LA.norm(r1,2);
            r1 = np.divide(r1,scale);
            #dist = LA.norm(r1-r0,2);
            # Note the formula to calculate the distance 
            eigVal = getApproxEigval(covMatrix,r1);
            dist = LA.norm(np.dot(covMatrix,r1)-eigVal*r1,2);
            #print dist;
            
            if dist < epsilon:
                #print count;
                #print "No.%d eigenvalue: %f" % (k,eigVal);
                break;
            else:    
                r0 = r1;
                count += 1;
        if eigVectors is None:
            eigVectors = r1;
        else:
            eigVectors = np.append(eigVectors,r1,axis=1);
        np.put(eigValues,k,eigVal);
        covMatrix -= eigVal*np.outer(r1,r1);
        k += 1;            
    return eigValues,eigVectors;


def simulatePrivateLocalPCA(data,maxDim,epsilon):
    k = np.minimum(maxDim,LA.matrix_rank(data));
    #print "In each data owner, the k is: %d" % k;
    
    WishartNoiseMatrix = DiffPrivImpl.SymmWishart(epsilon,data.shape[1]);
    C = np.cov(data,rowvar=False);
    noisyC = C + WishartNoiseMatrix;
    if data.shape[1]<100:
        noisyEigenvalues,noisyEigenvectors = LA.eig(noisyC);
        idx = np.absolute(noisyEigenvalues).argsort()[::-1];
        # print idx;
        noisyEigenvalues = np.real(noisyEigenvalues[idx]);
        # print sortedW;
        noisyEigenvectors = np.real(noisyEigenvectors[:, idx]);
    else:
        noisyEigenvalues,noisyEigenvectors = sparse.linalg.eigs(noisyC, k=k,tol=0.001);
    #noisyEigenvalues,noisyEigenvectors = genEigenvectors_power(noisyC, k);
    S = np.diagflat(np.sqrt(noisyEigenvalues));
    P = np.dot(noisyEigenvectors[:,:k],S[:k,:k]);
    return P;

def simulatePrivateGlobalPCA(data,numOfSamples,maxDim,epsilon):
    numOfCopies = data.shape[0]/numOfSamples;
    dataOwnerGroups = np.array_split(data, numOfCopies);
    P = None;
    for singleDataOwnerCopy in dataOwnerGroups:
        
        PPrime = simulatePrivateLocalPCA(singleDataOwnerCopy,maxDim,epsilon);
        if P is not None:
            k_prime = np.maximum(np.minimum(LA.matrix_rank(singleDataOwnerCopy),maxDim),LA.matrix_rank(P));
            tmpSummary = np.concatenate((PPrime, P), axis=0);
            P = simulatePrivateLocalPCA(tmpSummary,k_prime,epsilon);
        P = PPrime;
    return P;
def singleExp(xDimensions,trainingData,testingData,topK,isLinearSVM):
    
    pureTrainingData = trainingData[:,1:];
    trainingLabel = trainingData[:,0];
    #normalizedTrainingData = normByRow(pureTrainingData);
    
    pureTestingData = testingData[:,1:];
    testingLabel = testingData[:,0];
    #normalizedTestingData = normByRow(pureTestingData);
    scaler = StandardScaler(copy=False);
    #print pureTrainingData[0];
    scaler.fit(pureTrainingData);
    scaler.transform(pureTrainingData);
    #print pureTrainingData[0];

    #print pureTestingData[0];
    scaler.transform(pureTestingData);
    #print pureTestingData[0];

    numOfFeature = trainingData.shape[1]-1;
    cprResult = np.zeros((len(xDimensions),4));
    
    pcaImpl = PCAModule.PCAImpl(pureTrainingData);
    pcaImpl.getPCs(topK);
    
    '''
    To get a Wishart Noise projection matrix.
    '''
    WishartNoiseMatrix = DiffPrivImpl.SymmWishart(epsilon,numOfFeature);
    noisyCovMatrix = pcaImpl.covMatrix + WishartNoiseMatrix;
    #w, v = LA.eig(noisyCovMatrix);  
    # Sorting the eigenvalues in descending order.
    #idx = np.absolute(w).argsort()[::-1];
    #noisyProjMatrix = np.real(v[:,idx]);

    noisyEigValues,noisyProjMatrix = sparse.linalg.eigs(noisyCovMatrix, k=topK, tol=0.001);
    
    #print topK;
    pgProjMatrix = simulatePrivateGlobalPCA(pureTrainingData,numOfSamples,topK,epsilon);
    #print projTrainingData.shape;
    #for k in range(1,numOfDimensions):
    for k, targetDimension in np.ndenumerate(xDimensions):
        #print pcaImpl.projMatrix[:,0];
        cprResult[k][0] += targetDimension;
        projTrainingData1 = pcaImpl.transform(pureTrainingData,targetDimension);
        projTestingData1 = pcaImpl.transform(pureTestingData,targetDimension);
        print "Non-noise PCA %d" % targetDimension;
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData1,trainingLabel,projTestingData1,testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData1,trainingLabel,projTestingData1,testingLabel);
        
        cprResult[k][1] += result[2];
        
        projTrainingData2 = np.dot(pureTrainingData,noisyProjMatrix[:,:targetDimension]);
        projTestingData2 = np.dot(pureTestingData,noisyProjMatrix[:,:targetDimension]);
        
        print "DPDPCA %d" % targetDimension;
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData2,trainingLabel,projTestingData2,testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData2,trainingLabel,projTestingData2,testingLabel);
        
        cprResult[k][2] += result[2];


        projTrainingData3 = np.dot(pureTrainingData,pgProjMatrix[:,:targetDimension]);
        projTestingData3 = np.dot(pureTestingData,pgProjMatrix[:,:targetDimension]);
        #pgProjMatrix = simulatePrivateGlobalPCA(pureTrainingData,numOfSamples,targetDimension,epsilon);
        #projTrainingData3 = np.dot(pureTrainingData,pgProjMatrix);
        #projTestingData3 = np.dot(pureTestingData,pgProjMatrix);
        
        print "PrivateLocalPCA %d" % targetDimension;
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData3,trainingLabel,projTestingData3,testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData3,trainingLabel,projTestingData3,testingLabel);
        cprResult[k][3] += result[2];
        
    return cprResult;

def doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfDimensions,numOfSamples,isLinearSVM=True):
    data = np.loadtxt(datasetPath,delimiter=",");
    globalPCA = PCAModule.PCAImpl(data[:,1:]);
    numOfFeature = data.shape[1]-1;
    largestReducedFeature = globalPCA.getNumOfPCwithKPercentVariance(varianceRatio);
    print "%d/%d dimensions captures %.2f variance." % (largestReducedFeature,numOfFeature,varianceRatio);
    xDimensions = None;
    
    if numOfDimensions > numOfFeature:
        xDimensions = np.arange(1,numOfFeature);
        topK=numOfFeature;
    else:
        xDimensions = np.arange(1,largestReducedFeature,max(largestReducedFeature/numOfDimensions,1));
        topK=largestReducedFeature;
    #cprResult = np.zeros((len(xDimensions),4));
    cprResult = None;
    rs = ShuffleSplit(n_splits=numOfRounds, test_size=.2, random_state=0);
    rs.get_n_splits(data);
    
    #p = Pool(numOfRounds);
    normalizedData = gf.normByRow(data[:,1:]);

    normalizedData = np.concatenate((data[:,[0,]],normalizedData),axis=1);
    for train_index, test_index in rs.split(data):
        
        trainingData = normalizedData[train_index];
        testingData = normalizedData[test_index];
        #tmpResult = p.apply_async(singleExp, (xDimensions,trainingData,testingData,topK,isLinearSVM));
        #cprResult += tmpResult.get();
        tmpResult = singleExp(xDimensions, trainingData, testingData, topK, isLinearSVM);
        if cprResult is None:
            cprResult = tmpResult;
        else:
            cprResult = np.concatenate((cprResult,tmpResult),axis=0);
        """
        for i in range(0,len(cprResult)):
            print ','.join(['%.3f' % num for num in cprResult[i]]);
        """
    #avgResult = cprResult/numOfRounds;
    avgResult = cprResult;
    #p.close();
    #p.join();
    for result in avgResult:
        print ','.join(['%.3f' % num for num in result]);

    return avgResult;

if __name__ == "__main__":
    
    numOfRounds = 10;
    epsilon = 0.3;
    varianceRatio = 0.9
    numOfSamples = 2;
    numOfDimensions = 30;
    figSavedPath = "./fig/";
    resultSavedPath = "./log/";
    isLinearSVM = False;
    if len(sys.argv) > 1:
        datasetPath = sys.argv[1];
        print "+++ using passed in arguments: %s" % (datasetPath);
        result = doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfDimensions,numOfSamples,isLinearSVM=isLinearSVM);
        np.savetxt(resultSavedPath+"dataOwner_"+os.path.basename(datasetPath)+".output",result,delimiter=",",fmt='%1.3f');
    else:
        datasets = ['CNAE','YaleB','p53 Mutant','CNAE_5','CNAE_7','face2','Amazon_3','madelon'];
        for dataset in datasets:
            print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
            datasetPath = "./input/"+dataset+"_prePCA";
            timeStr = str(time());
            #result = doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfDimensions,numOfSamples,isLinearSVM=isLinearSVM);
            #np.savetxt(resultSavedPath+"dataOwner_"+dataset+".output",result,delimiter=",",fmt='%1.3f');
            drawF1Score(dataset,data=None,path = resultSavedPath+"dataOwner_"+dataset+".output",figSavedPath=figSavedPath);