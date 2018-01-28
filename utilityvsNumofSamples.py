from pkg.svm import SVMModule;
from pkg.dimReduction import PCAModule;
#from pkg.DPDPCA.DataOwner import DataOwnerImpl;
import numpy as np;
from numpy import linalg as LA;
from pkg.diffPrivDimReduction import invwishart;
from numpy.linalg import norm;
from sklearn.model_selection import ShuffleSplit;
from sklearn.model_selection import StratifiedShuffleSplit;
from pkg.diffPrivDimReduction.DPModule import DiffPrivImpl;
import sys;
import os;
from multiprocessing import Pool;
import scipy.sparse as sparse;
from sklearn.preprocessing import StandardScaler;
from pkg.global_functions import globalFunction as gf;

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


def simulatePrivateLocalPCA(data,targetDimension,epsilon):
    #k = np.minimum(targetDimension,LA.matrix_rank(data));
    #print "In each data owner, the k is: %d" % k;

    C = data.T.dot(data);
    WishartNoiseMatrix = DiffPrivImpl.SymmWishart(epsilon,data.shape[1]);
    noisyC = C + WishartNoiseMatrix;
    """
    if data.shape[1]<100:
        noisyEigenvalues,noisyEigenvectors = LA.eig(noisyC);
        idx = np.absolute(noisyEigenvalues).argsort()[::-1];
        # print idx;
        noisyEigenvalues = noisyEigenvalues[idx];
        # print sortedW;
        noisyEigenvectors = noisyEigenvectors[:, idx];
    else:
        noisyEigenvalues,noisyEigenvectors = sparse.linalg.eigs(noisyC, k=max(k-1,1),tol=0.001);
    #noisyEigenvalues,noisyEigenvectors = genEigenvectors_power(noisyC, k);
    """
    noisyLeftSigVectors,noisySingularValues, noisyRightSigVectors = sparse.linalg.svds(noisyC, k=targetDimension, tol=0.001);
    noisySingularValues = np.real(noisySingularValues);
    noisyRightSigVectors = np.real(noisyRightSigVectors.T);
    S = np.diagflat(np.sqrt(noisySingularValues));
    P = np.dot(noisyRightSigVectors,S);
    return P;

def simulatePrivateGlobalPCA(data,numOfSamples,targetDimension,epsilon):
    numOfCopies = data.shape[0]/numOfSamples;
    dataOwnerGroups = np.array_split(data, numOfCopies);
    P = None;
    for singleDataOwnerCopy in dataOwnerGroups:
        k_prime = min(targetDimension,LA.matrix_rank(singleDataOwnerCopy));
        P_prime = simulatePrivateLocalPCA(singleDataOwnerCopy, k_prime, epsilon);
        if P is not None:
            k_prime = max(k_prime, LA.matrix_rank(P));
            tmpSummary = np.hstack((P_prime, P));
            P = simulatePrivateLocalPCA(tmpSummary.T, k_prime, epsilon);
        else:
            P = P_prime;
    return P;

def singleExp(xSamples,trainingData,testingData,topK,epsilon,isLinearSVM):
    
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
    cprResult = []
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
    noisyProjMatrix = np.real(noisyProjMatrix);

    # Project the data using different projection matrix.
    projTrainingData1 = pcaImpl.transform(pureTrainingData, topK);
    projTestingData1 = pcaImpl.transform(pureTestingData, topK);
    print "Non-noise PCA %d" % topK;
    if isLinearSVM:
        result = SVMModule.SVMClf.linearSVM(projTrainingData1, trainingLabel, projTestingData1, testingLabel);
    else:
        result = SVMModule.SVMClf.rbfSVM(projTrainingData1, trainingLabel, projTestingData1, testingLabel);

    cprResult.append(result[2]);

    projTrainingData2 = np.dot(pureTrainingData, noisyProjMatrix);
    projTestingData2 = np.dot(pureTestingData, noisyProjMatrix);

    print "DPDPCA %d" % topK;
    if isLinearSVM:
        result = SVMModule.SVMClf.linearSVM(projTrainingData2, trainingLabel, projTestingData2, testingLabel);
    else:
        result = SVMModule.SVMClf.rbfSVM(projTrainingData2, trainingLabel, projTestingData2, testingLabel);

    cprResult.append(result[2]);

    for k, numOfSamples in np.ndenumerate(xSamples):
        pgProjMatrix = simulatePrivateGlobalPCA(pureTrainingData,numOfSamples,topK,epsilon);
        #print projTrainingData.shape;

        #print pcaImpl.projMatrix[:,0];

        projTrainingData3 = np.dot(pureTrainingData,pgProjMatrix);
        projTestingData3 = np.dot(pureTestingData,pgProjMatrix);
        
        print "PrivateLocalPCA with %d data held by each data owner" % numOfSamples;
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData3,trainingLabel,projTestingData3,testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData3,trainingLabel,projTestingData3,testingLabel);
        cprResult.append(result[2]);
        
    return np.asarray(cprResult);

def doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfPointsinXAxis, isLinearSVM=True):
    if os.path.basename(datasetPath).endswith('npy'):
        data = np.load(datasetPath);
    else:
        data = np.loadtxt(datasetPath, delimiter=",");
    scaler = StandardScaler();
    data_std = scaler.fit_transform(data[:, 1:]);
    globalPCA = PCAModule.PCAImpl(data_std);
    largestReducedFeature = globalPCA.getNumOfPCwithKPercentVariance(varianceRatio);

    numOfFeature = data.shape[1]-1;
    print "%d/%d dimensions captures %.2f variance." % (largestReducedFeature,numOfFeature,varianceRatio);
    #cprResult = np.zeros((len(xDimensions),4));
    cprResult = None;
    rs = StratifiedShuffleSplit(n_splits=numOfRounds, test_size=.2, random_state=0);
    rs.get_n_splits(data[:,1:],data[:,0]);
    
    #p = Pool(numOfRounds);
    normalizedData = gf.normByRow(data[:,1:]);
    normalizedData = np.concatenate((data[:,[0,]],normalizedData),axis=1);

    for train_index, test_index in rs.split(data[:,1:],data[:,0]):

        trainingData = normalizedData[train_index];
        testingData = normalizedData[test_index];
        print "number of training samples %d" % trainingData.shape[0];
        #tmpResult = p.apply_async(singleExp, (xDimensions,trainingData,testingData,topK,isLinearSVM));
        #cprResult += tmpResult.get();
        mostSamplesPerDataOwner = trainingData.shape[0] / 2;
        xSamples = np.arange(2, mostSamplesPerDataOwner, max(mostSamplesPerDataOwner/numOfPointsinXAxis, 1));
        print "number of samples be tested: %s" % xSamples;
        tmpResult = singleExp(xSamples, trainingData, testingData, largestReducedFeature, epsilon, isLinearSVM);
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

def doExp_unbalanced(datasetPath,epsilon,varianceRatio,numOfRounds,numOfPointsinXAxis,isLinearSVM=True):
    if os.path.basename(datasetPath).endswith('npy'):
        data = np.load(datasetPath);
    else:
        data = np.loadtxt(datasetPath, delimiter=",");

    numOfFeature = data.shape[1] - 1;

    scaler = StandardScaler(copy=False);
    # print pureTrainingData[0];
    rawData = data[:, 1:];
    scaler.fit(rawData);
    scaler.transform(rawData);
    globalPCA = PCAModule.PCAImpl(rawData);
    largestReducedFeature = globalPCA.getNumOfPCwithKPercentVariance(varianceRatio);
    print "%d/%d dimensions captures %.2f variance." % (largestReducedFeature, numOfFeature, varianceRatio);
    xDimensions = None;

    normalizedData = gf.normByRow(data[:, 1:]);
    normalizedData = np.concatenate((data[:, [0, ]], normalizedData), axis=1);
    posiData = data[np.where(normalizedData[:,0]==1)];
    negData = data[np.where(normalizedData[:,0]==-1)];
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
        mostSamplesPerDataOwner = trainingData.shape[0] / 2;
        xSamples = np.arange(2, mostSamplesPerDataOwner, max(mostSamplesPerDataOwner / numOfPointsinXAxis, 1));
        print "number of samples be tested: %s" % xSamples;
        tmpResult = singleExp(xSamples, trainingData, testingData, largestReducedFeature, epsilon, isLinearSVM);
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
    
    numOfRounds = 3;
    epsilon = 0.3;
    varianceRatio = 0.8
    #numOfSamples = 2;
    numOfPointsinXAxis = 20;
    figSavedPath = "./log/";
    resultSavedPath = "./log/";
    isLinearSVM = False;
    if len(sys.argv) > 1:
        datasetPath = sys.argv[1];
        print "+++ using passed in arguments: %s" % (datasetPath);
        #result = doExp_unbalanced(datasetPath,epsilon,varianceRatio,numOfRounds,numOfPointsinXAxis,isLinearSVM=isLinearSVM);
        result = doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfPointsinXAxis,isLinearSVM=isLinearSVM);
        np.savetxt(resultSavedPath+"dataOwner_"+os.path.basename(datasetPath)+".output",result,delimiter=",",fmt='%1.3f');
    else:
        datasets = ['Diabetes','CNAE_2','Face_15','p53_3000'];
        for dataset in datasets:
            print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
            datasetPath = "./input/"+dataset+"_prePCA";
            result = doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfPointsinXAxis,isLinearSVM=isLinearSVM);
            np.savetxt(resultSavedPath+"dataOwner_"+dataset+".output",result,delimiter=",",fmt='%1.3f');
