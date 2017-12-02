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

def drawF1Score(datasetTitle,data=None,path=None,figSavedPath=None):
    plt.clf();
    if path is not None:
        data = np.loadtxt(path,delimiter=",");
    xBound = len(data)+1;
    x = range(1,xBound);
    minVector = np.amin(data,axis=0);
    yMin = min(minVector);
    maxVector = np.amax(data,axis=0);
    yMax = max(maxVector);
    
    yMin = (yMin-0.1) if (yMin-0.1)>0 else 0;
    yMax = (yMax+0.1) if (yMax+0.1)<1 else 1;
    #x = [10,40,70,100,130,160,190,220,250,280,310,340];
    y1Line,y2Line,y3Line = plt.plot(x, data[:,2], 'bo-', x, data[:,5], 'r^-',x, data[:,8], 'gs-');
    if datasetTitle is 'Ionosphere':
        plt.legend([y1Line,y2Line,y3Line], ['PCA','DPDPCA','PrivateLocalPCA'],loc=4);
    else:
        plt.legend([y1Line,y2Line,y3Line], ['PCA','DPDPCA','PrivateLocalPCA'],loc=2);
    plt.axis([0,xBound,yMin,yMax]);
    #plt.axis([0,10,0.4,1.0]);
    plt.xlabel('Number of Principal Components',fontsize=18);
    plt.ylabel('F1-Score',fontsize=18);
    plt.title(datasetTitle+' Dataset', fontsize=18);
    plt.xticks(x);
    if figSavedPath is None:
        plt.show();
    else:
        plt.savefig(figSavedPath+"twoSamplesatDataOwner_"+datasetTitle+'.pdf', format='pdf', dpi=1000);
def simulatePrivateLocalPCA(data,maxDim,epsilon):
    k = np.minimum(maxDim,LA.matrix_rank(data));
    #print "In each data owner, the k is: %d" % k;
    
    WishartNoiseMatrix = DiffPrivImpl.SymmWishart(epsilon,data.shape[1]);
    C = np.dot(data.T,data);
    noisyC = C + WishartNoiseMatrix;
    U, s, V = np.linalg.svd(noisyC);
    
    S = np.diagflat(np.sqrt(s));
    P = np.dot(U[:,0:k],S[0:k,0:k]);
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

def doExp(datasetPath,numOfRounds,epsilon,numOfSamples,isLinearSVM=True):
    data = np.loadtxt(datasetPath,delimiter=",");
    rs = ShuffleSplit(n_splits=numOfRounds, test_size=.2, random_state=0);
    rs.get_n_splits(data);
    
    numOfFeature = data.shape[1]-1;
    
    cprResult = np.zeros((numOfFeature-1,9));
    epsilon = 0.5;
    
    for train_index, test_index in rs.split(data):
        
        trainingData = data[train_index];
        pureTrainingData = trainingData[:,1:];
        trainingLabel = trainingData[:,0];
        normalizedTrainingData = normByRow(pureTrainingData);
        
        numOfFeature = trainingData.shape[1]-1;
        
        testingData = data[test_index];
        pureTestingData = testingData[:,1:];
        testingLabel = testingData[:,0];
        normalizedTestingData = normByRow(pureTestingData);
        
        pcaImpl = PCAModule.PCAImpl(normalizedTrainingData);
        pcaImpl.getPCs();
        
        '''
        To get a Wishart Noise projection matrix.
        '''
        WishartNoiseMatrix = DiffPrivImpl.SymmWishart(epsilon,numOfFeature);
        noisyCovMatrix = pcaImpl.covMatrix + WishartNoiseMatrix;
        w, v = LA.eig(noisyCovMatrix);  
        # Sorting the eigenvalues in descending order.
        idx = np.absolute(w).argsort()[::-1];
        noisyProjMatrix = np.real(v[:,idx]);
        
        #print projTrainingData.shape;
    
        for k in range(1,numOfFeature):
            #print pcaImpl.projMatrix[:,0];
            print "Features %d:" % (k);   
            projTrainingData1 = pcaImpl.transform(normalizedTrainingData,k);
            projTestingData1 = pcaImpl.transform(normalizedTestingData,k);
        
            print "-PCA SVM training";
            if isLinearSVM:
                result = SVMModule.SVMClf.linearSVM(projTrainingData1,trainingLabel,projTestingData1,testingLabel);
            else:
                result = SVMModule.SVMClf.rbfSVM(projTrainingData1,trainingLabel,projTestingData1,testingLabel);
            
            cprResult[k-1][0] = cprResult[k-1][0]+result[0];
            cprResult[k-1][1] = cprResult[k-1][1]+result[1];
            cprResult[k-1][2] = cprResult[k-1][2]+result[2];
            
            
            projTrainingData2 = np.dot(normalizedTrainingData,noisyProjMatrix[:,:k]);
            projTestingData2 = np.dot(normalizedTestingData,noisyProjMatrix[:,:k]);
            
            print "My Wishart-DPPCA SVM training";
            if isLinearSVM:
                result = SVMModule.SVMClf.linearSVM(projTrainingData2,trainingLabel,projTestingData2,testingLabel);
            else:
                result = SVMModule.SVMClf.rbfSVM(projTrainingData2,trainingLabel,projTestingData2,testingLabel);
            
            cprResult[k-1][3] = cprResult[k-1][3]+result[0];
            cprResult[k-1][4] = cprResult[k-1][4]+result[1];
            cprResult[k-1][5] = cprResult[k-1][5]+result[2];
            
            pgProjMatrix = simulatePrivateGlobalPCA(normalizedTrainingData,numOfSamples,k,epsilon);
            projTrainingData3 = np.dot(normalizedTrainingData,pgProjMatrix);
            projTestingData3 = np.dot(normalizedTestingData,pgProjMatrix);
            
            print "PrivateLocalPCA SVM training";
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

def normByRow(data):
    for i in range(0,data.shape[0]):
        rowNorm = norm(data[i,:], ord=2);
        data[i,:] = data[i,:]/rowNorm;
    return data;
"""
def privateGlobalPCA(folderPath,k,epsilon):
    # Get the folder, which contains all the horizontal data.
    dataFileList = glob.glob(folderPath+"/*");
    data = np.loadtxt(dataFileList[0],delimiter=",");
    # Here, it should be OK with data[:,1:data.shape[1]];
    matrix = data[:,1:data.shape[1]];
    #k = matrix.shape[1]-1;
    #k = 5;
    print k;
    dataOwner = DataOwnerImpl(dataFileList[0]);
    P = dataOwner.privateLocalPCA(None,k,epsilon);
    #print P.shape;
    dataFileList.pop(0);
    #print "Private Global PCA computing:";

    for path in dataFileList:
        #print str(int(round(time.time() * 1000)))+", "+path;
        dataOwner = DataOwnerImpl(path);
        PPrime = dataOwner.privateLocalPCA(None,k,epsilon);
        
        k_prime = np.maximum(np.minimum(LA.matrix_rank(dataOwner.data),k),LA.matrix_rank(P));
        
        tmpSummary = np.concatenate((PPrime, P), axis=1);
        
        P = dataOwner.privateLocalPCA(tmpSummary.T,k_prime,epsilon);
        #print tmpSummary.shape; 

    #print P[:,0];
    return P;
def globalPCA_NoNoise(folderPath):
    dataFileList = glob.glob(folderPath+"/*");
    data = np.loadtxt(dataFileList[0],delimiter=",");
    sumR = None;
    sumV = None;
    sumN = 0;
    for path in dataFileList:
        dataOwner = DataOwnerImpl(path);
        R = np.dot(dataOwner.data.T,dataOwner.data);       
        v = np.sum(dataOwner.data,axis=0);
        N = dataOwner.data.shape[0];
        if sumR is None:
            sumR = R;
            sumV = v;
        else:
            sumR = sumR + R;
            sumV = sumV + v;
            sumN = sumN + N;
            
    ScatterMatrix = sumR - np.divide(np.outer(v,v),sumN);
    U, s, V = np.linalg.svd(ScatterMatrix);
    
    S = np.diagflat(np.sqrt(s));
    #print U.dot(S)[:,0];
    return U.dot(S);

def myGlobalPCA(folderPath,epsilon):
    dataFileList = glob.glob(folderPath+"/*");
    data = np.loadtxt(dataFileList[0],delimiter=",");
    sumR = None;
    sumV = None;
    sumN = 0;
    for path in dataFileList:
        dataOwner = DataOwnerImpl(path);
        R = np.dot(dataOwner.data.T,dataOwner.data);       
        v = np.sum(dataOwner.data,axis=0);
        N = dataOwner.data.shape[0];
        if sumR is None:
            sumR = R;
            sumV = v;
        else:
            sumR = sumR + R;
            sumV = sumV + v;
            sumN = sumN + N;
            
    ScatterMatrix = sumR - np.divide(np.outer(v,v),sumN);
    
    df = len(ScatterMatrix)+1;
    sigma = 1/epsilon*np.identity(len(ScatterMatrix));
    #print sigma;
    wishart = invwishart.wishartrand(df,sigma);

    U, s, V = np.linalg.svd(ScatterMatrix+wishart);
    
    '''
    To double check if 1) the EVD gets the correct value, 2) the same as the SVD result.
    w, v = LA.eig(S);    
    # Sorting the eigenvalues in descending order.
    idx = np.absolute(w).argsort()[::-1];
    #print idx;
    sortedW = w[idx];
    #print sortedW;
    sortedV = v[:,idx];
    print sortedV[:,0];
    '''
    #U, s, V = LA.svd(S);
    S = np.diagflat(np.sqrt(s));
    #print U.dot(S)[:,0];
    return U.dot(S);
"""
if __name__ == "__main__":
    
    datasets = ['diabetes','australian','german', 'ionosphere', 'madelon'];
    numOfRounds = 10;
    epsilon = 0.5;
    numOfSamples = 2;
    figSavedPath = "./log/";
    for dataset in datasets:
        print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
        datasetPath = "../distr_dp_pca/experiment/input/"+dataset+"_prePCA";
        result = doExp(datasetPath,numOfRounds,epsilon,numOfSamples,isLinearSVM=True);
        drawF1Score(dataset,result,figSavedPath=figSavedPath);
        """
        outputFolderPath = datasetPath+"_referPaper2/plaintext/";
        trainingDataPath = datasetPath+"_training";
        testingDataPath = datasetPath+"_testing";
        #for i in range(10):
        
        cprResult = np.zeros((11,3));
        totalRound = 10;
        epsilon = 0.5;
        
        for j in range(totalRound):
            gf.genTrainingTestingData(datasetPath,trainingDataPath,testingDataPath);
            
            trainingData = np.loadtxt(trainingDataPath,delimiter=",");
            pureTrainingData = trainingData[:,1:];
            trainingLabel = trainingData[:,0];
            pureTrainingData = normByRow(pureTrainingData);
            
            #numOfFeature = trainingData.shape[1];
            #numOfDataPerOwner = np.floor(numOfFeature/5.0);
            #numOfTrunks = int(np.ceil(trainingData.shape[0]/numOfDataPerOwner));
            numOfTrunks = int(np.ceil(trainingData.shape[0]/2));
            print "The number of trunks is: %d" % numOfTrunks;
            gf.splitAndSaveDatasets(trainingDataPath,outputFolderPath,numOfTrunks);
            
            #pcaImpl = PCAModule.PCAImpl(pureTrainingData);
            #print pcaImpl.projMatrix[:,0];
            
            testingData = np.loadtxt(testingDataPath,delimiter=",");
            pureTestingData = testingData[:,1:];
            pureTestingData = normByRow(pureTestingData);
            #trainingColMean = np.mean(pureTrainingData,axis=0);
            #trainingColDeviation = np.std(pureTrainingData, axis=0);
            
            #scaledTrainingData = np.divide((pureTrainingData - trainingColMean),trainingColDeviation);
            #scaledTestingData = np.divide((pureTestingData - trainingColMean),trainingColDeviation);
            
            testingLabel = testingData[:,0];
            projMatrix = myGlobalPCA(outputFolderPath,epsilon);
            projMatrix_NoNoise = globalPCA_NoNoise(outputFolderPath);
            #print projMatrix[:,0:10].shape;
            
            #result = SVMModule.SVMClf.rbfSVM(pureTrainingData,trainingLabel,pureTestingData,testingLabel);
            #print result;
            
            for k in range(1,11):
                
                kProjMatrix_NoNoise = projMatrix_NoNoise[:,0:k];
                projTrainingData = np.dot(pureTrainingData,kProjMatrix_NoNoise);
                projTestingData = np.dot(pureTestingData,kProjMatrix_NoNoise);
                #print projTestingData.shape;
                result = SVMModule.SVMClf.rbfSVM(projTrainingData,trainingLabel,projTestingData,testingLabel);
                cprResult[k-1][0] = cprResult[k-1][0]+result[2];
                
                
                pgProjMatrix = privateGlobalPCA(outputFolderPath,k,epsilon); 
                #print pgProjMatrix.shape;   
                projTrainingData = np.dot(pureTrainingData,pgProjMatrix);
                projTestingData = np.dot(pureTestingData,pgProjMatrix);
                #print projTrainingData.shape;
                
                result = SVMModule.SVMClf.rbfSVM(projTrainingData,trainingLabel,projTestingData,testingLabel);
                cprResult[k-1][1] = cprResult[k-1][1]+result[2];
                
                kProjMatrix = projMatrix[:,0:k];
                projTrainingData = np.dot(pureTrainingData,kProjMatrix);
                projTestingData = np.dot(pureTestingData,kProjMatrix);
                #print projTestingData.shape;
                result = SVMModule.SVMClf.rbfSVM(projTrainingData,trainingLabel,projTestingData,testingLabel);
                cprResult[k-1][2] = cprResult[k-1][2]+result[2];
                
                print "===========================";
            for i in range(0,len(cprResult)):
                print "%f,%f,%f" % (cprResult[i][0],cprResult[i][1],cprResult[i][2]);
        
        print "******************************";
        for i in range(0,len(cprResult)):
            print "%f,%f,%f" % (cprResult[i][0]/totalRound,cprResult[i][1]/totalRound,cprResult[i][2]/totalRound);
        """