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

def privateGlobalPCA(folderPath,k):
    
    epsilon = 0.6;
    # Get the folder, which contains all the horizontal data.
    dataFileList = glob.glob(folderPath+"/*");
    data = np.loadtxt(dataFileList[0],delimiter=",");
    # Here, it should be OK with data[:,1:data.shape[1]];
    matrix = data[:,range(1,data.shape[1])];
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
    
def myGlobalPCA(folderPath):
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
            
    S = sumR - np.divide(np.outer(v,v),sumN);
    
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
    U, s, V = LA.svd(S);
    S = np.diagflat(np.sqrt(s));
    #print U.dot(S)[:,0];
    return U.dot(S);

if __name__ == "__main__":
    
    dataset = "australian";
    datasetPath = "../distr_dp_pca/experiment/input/"+dataset+"_prePCA";
    outputFolderPath = datasetPath+"_referPaper2/plaintext/";
    trainingDataPath = datasetPath+"_training";
    testingDataPath = datasetPath+"_testing";
    #for i in range(10):
    
    gf.genTrainingTestingData(datasetPath,trainingDataPath,testingDataPath);
    numOfTrunks = 20;
    gf.splitAndSaveDatasets(trainingDataPath,outputFolderPath,numOfTrunks);


    trainingData = np.loadtxt(trainingDataPath,delimiter=",");
    pureTrainingData = trainingData[:,1:];
    trainingLabel = trainingData[:,0];
    
    #pcaImpl = PCAModule.PCAImpl(pureTrainingData);
    #print pcaImpl.projMatrix[:,0];
    
    testingData = np.loadtxt(testingDataPath,delimiter=",");
    pureTestingData = testingData[:,1:];
    testingLabel = testingData[:,0];
    projMatrix = myGlobalPCA(outputFolderPath);
    pgResultList=[];
    myResultList=[];
    
    
    #result = SVMModule.SVMClf.rbfSVM(pureTrainingData,trainingLabel,pureTestingData,testingLabel);
    #print result;
    
    for k in range(1,10):
        pgProjMatrix = privateGlobalPCA(outputFolderPath,k); 
        #print pgProjMatrix.shape;   
        projTrainingData = np.dot(pureTrainingData,pgProjMatrix);
        projTestingData = np.dot(pureTestingData,pgProjMatrix);
        
        result = SVMModule.SVMClf.rbfSVM(projTrainingData,trainingLabel,projTestingData,testingLabel);
        pgResultList.append(result);
        
        kProjMatrix = projMatrix[:,0:k];
        projTrainingData = np.dot(pureTrainingData,kProjMatrix);
        projTestingData = np.dot(pureTestingData,kProjMatrix);
        result = SVMModule.SVMClf.rbfSVM(projTrainingData,trainingLabel,projTestingData,testingLabel);
        myResultList.append(result);
        print "===========================";

    for i in range(0,len(pgResultList)):
        print "%f , %f" % (pgResultList[i][2],myResultList[i][2]);
    