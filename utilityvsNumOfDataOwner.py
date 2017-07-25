from pkg.global_functions import globalFunction as gf;
from pkg.svm import SVMModule;
from pkg.dimReduction import PCAModule;

import timeit;
import os;
import sys;
from pkg.DPDPCA.DataOwner import DataOwnerImpl;
import numpy as np;
import glob;
from numpy import linalg as LA;
from pkg.wishart import invwishart;
from sklearn import svm;
from numpy.linalg import norm;

def normByRow(data):
    for i in range(0,data.shape[0]):
        rowNorm = norm(data[i,:], ord=2);
        data[i,:] = data[i,:]/rowNorm;
    return data;

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

if __name__ == "__main__":
    
    datasets = ['diabetes','australian','german', 'ionosphere', 'madelon'];
    
    for dataset in datasets:
        print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
        datasetPath = "../distr_dp_pca/experiment/input/"+dataset+"_prePCA";
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
        