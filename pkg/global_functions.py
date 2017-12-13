#!/usr/bin/env python
import numpy as np;
from numpy import linalg as LA;
from pkg.diffPrivDimReduction import invwishart;
import os;
from numpy.linalg import norm;

class globalFunction(object):
    
    @classmethod
    def normByRow(cls,data):
        rowsNorm = LA.norm(data, axis=1);
        maxL2Norm = np.amax(rowsNorm);
        """
        for i in range(data.shape[0]):
            rowNorm = norm(data[i,:], ord=2);
            data[i,:] = data[i,:]/rowNorm;
        """
        return data/maxL2Norm;

    '''
    def distance2(covMatrix,r1):
        
        #Approximate the eigenvalue.
        temp1 = np.dot(covMatrix,r1);
        v1 = np.dot(r1.T,temp1);
        v2 = np.dot(r1.T,r1);
        eigVal = np.divide(v1,v2);
        
        #Return the difference between M*r1 and lambda*r1;
        return LA.norm(temp1-eigVal*r1,2);
    
   
    def distance(v1, v2):
        v = v1 - v2;
        v = v * v;
        return np.sum(v);
  
    '''

    @classmethod
    def genTrainingTestingData(cls,dataPath,trainingFilePath,testingFilePath):
        data = np.loadtxt(dataPath,delimiter=",");
        #print data.shape[0];
        shuffleData = np.random.permutation(data);
        testIndex = -shuffleData.shape[0]/4;
        testMatrix = shuffleData[testIndex:-1,:];
        np.savetxt(testingFilePath,testMatrix,delimiter=",",fmt='%1.7f');
        numOfPositive = 0;
        numOfNegative = 0
        for i in range(0,testMatrix.shape[0]):
            if testMatrix[i,0]>0:
                numOfPositive = numOfPositive + 1;
            else:
                numOfNegative = numOfNegative + 1;
        print "Number of testing samples: "+ str(testMatrix.shape[0]);
        print "Number of positive samples: " + str(numOfPositive);
        print "Number of negative samples: " + str(numOfNegative);
        #print testMatrix.shape[0];
        #print testMatrix;
        trainMatrix = shuffleData[:(shuffleData.shape[0]+testIndex),:];
        
        numOfPositive = 0;
        numOfNegative = 0
        for i in range(0,trainMatrix.shape[0]):
            if trainMatrix[i,0]>0:
                numOfPositive = numOfPositive + 1;
            else:
                numOfNegative = numOfNegative + 1;
        print "Number of training samples: "+ str(trainMatrix.shape[0]);
        print "Number of positive samples: " + str(numOfPositive);
        print "Number of negative samples: " + str(numOfNegative);
        
        np.savetxt(trainingFilePath,trainMatrix,delimiter=",",fmt='%1.7f');
        #print trainMatrix.shape[0];
        return;

    @classmethod
    def splitAndSaveDatasets(cls,inputFilePath,outputFolderPath,numOfTrunks):
        data = np.loadtxt(inputFilePath,delimiter=",");
        shuffleData = np.random.permutation(data);
        
        if not os.path.exists(outputFolderPath):
            os.system('mkdir -p %s' % outputFolderPath);
        
        subDataSets = np.array_split(shuffleData,numOfTrunks);
        for i in range(0,numOfTrunks):
            np.savetxt(outputFolderPath+"/"+str(i),subDataSets[i],delimiter=",",fmt='%1.2f');
        '''
        My version of split, don't invent the wheels again. so, later, i find the array_split() methods, which 
        allows the numOfTrunkss to be an integer that does not equally divide the axis. 
        binSize = shuffleData.shape[0]/numOfTrunks;
        for i in range(0,numOfTrunks):
            np.savetxt(outputFolderPath+"/"+str(i),shuffleData[(i-1)*binSize:i*binSize-1,:],delimiter=",",fmt='%1.7f');
        '''
    @classmethod
    def calcMeanandStd(cls,data):
        tmpMean = np.mean(data, axis=0);
        tmpStd = np.std(data, axis=0);
        return tmpMean, tmpStd;
