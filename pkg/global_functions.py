#!/usr/bin/env python

import numpy as np;
from numpy import linalg as LA;
from wishart import invwishart;
import os;
from numpy.linalg import norm;

class globalFunction(object):
    @classmethod
    def normByRow(cls,data):
        
        for i in range(0,data.shape[0]):
            rowNorm = norm(data[i,:], ord=2);
            data[i,:] = data[i,:]/rowNorm;
        return data;
    
    @classmethod
    def genCovMatrix(cls,trainingDataPath):
        data = np.loadtxt(trainingDataPath,delimiter=",");
        matrix = data[:,range(1,data.shape[1])];
        
        # Scaling data.
        cMax = np.amax(matrix,axis=0);
        cMin = np.amin(matrix,axis=0);
        matrix = np.divide((matrix - cMin),(cMax-cMin));
        
    # It seems that I don't need to subtract the mean vector, which is used to calculate for the scatter matrix,
    # but here, since I am using the np.cov(), I can just throw the raw matrix in.
    #    columnMean = np.mean(matrix,axis=0);
    #    for k in range(len(matrix)):
    #        matrix[k,:]=matrix[k,:]-columnMean;
        covMatrix = np.cov(matrix.T);
        return cMin,cMax,covMatrix;
    @classmethod
    def genEigenvectors(cls,covMatrix):
        w, v = LA.eig(covMatrix);
        
        # Sorting the eigenvalues in descending order.
        idx = np.absolute(w).argsort()[::-1];
        #print idx;
        sortedW = w[idx];
        #print sortedW;
        sortedV = v[:,idx];
    
        return sortedW,sortedV;
    @classmethod
    def getApproxEigval(cls,covMatrix,r1):
        temp1 = np.dot(covMatrix,r1);
        v1 = np.dot(r1.T,temp1);
        v2 = np.dot(r1.T,r1);
        eigVal = np.divide(v1,v2);
        return eigVal;
    
    @classmethod
    def genEigenvectors_power(cls,covMatrix):
    #    r0 = np.random.rand(covMatrix.shape[0],1);
        epsilon = 0.01;
        eigVectors = [];
        k=0;
        while k<covMatrix.shape[0]:
            r0 = np.random.rand(covMatrix.shape[0],1);
            count=0;
            while True:
                r1 = np.dot(covMatrix, r0);
                # Get the second norm of r1;
                scale = LA.norm(r1,2);
                r1 = np.divide(r1,scale);
                #dist = LA.norm(r1-r0,2);
                eigVal = getApproxEigval(covMatrix,r1);
                dist = LA.norm(np.dot(covMatrix,r1)-eigVal*r1,2);
                #print dist;
                
                if dist < epsilon:
                    #print count;
                    #print eigVal;
                    break;
                else:    
                    r0 = r1;
                    count = count + 1;
            #print (r1.dot(r1.T)); 
            eigVectors.append(r1);
            covMatrix = covMatrix - covMatrix.dot(r1.dot(r1.T));
            k = k+1;            
        return np.asarray(eigVectors).T;
    
    '''
    def distance2(covMatrix,r1):
        
        #Approximate the eigenvalue.
        temp1 = np.dot(covMatrix,r1);
        v1 = np.dot(r1.T,temp1);
        v2 = np.dot(r1.T,r1);
        eigVal = np.divide(v1,v2);
        
        #Return the difference between M*r1 and lambda*r1;
        return LA.norm(temp1-eigVal*r1,2);
    
    def genEigenvectors_power(covMatrix):
        r0 = np.random.rand(covMatrix.shape[0],1);
    #    r0 = np.array([[1],[1]]);
        epsilon = 0.01;
        count=0;
        while True:
                r1 = np.dot(covMatrix, r0);
                # Get the second norm of r1;
                scale = LA.norm(r1,2);
                r1 = np.divide(r1,scale);
                #dist = LA.norm(r1-r0,2);
                dist = distance2(covMatrix,r1);
                #print dist;
                
                if dist < epsilon:
                    print count;
                    break;
                else:    
                    r0 = r1;
                    count = count + 1;
        return r1;
    
    def distance(v1, v2):
        v = v1 - v2;
        v = v * v;
        return np.sum(v);
        
    def genEigenvectors_power(covMatrix):
        r0 = np.random.rand(covMatrix.shape[0],1);
        epsilon = 0.1;
        while True:
                r1 = np.dot(covMatrix, r0);
                dist = distance(r1, r0);
                if dist < epsilon:
                    break;
                else:
                    r0 = r1;
        return r1;
    '''
    @classmethod
    def genProjMatrix(cls,eigenvectors,reducedFeature):
        projMatrix = eigenvectors[:,0:reducedFeature];
        return projMatrix;
    
    @classmethod
    def genNoisyEigenvectors(cls,covMatrix):
        df = len(covMatrix)+1;
        sigma = 1/0.6*np.identity(len(covMatrix));
        #print sigma;
        wishart = invwishart.wishartrand(df,sigma);
        #print wishart;
        return genEigenvectors(covMatrix+wishart);
    
    @classmethod
    def genProjData(cls,inputFilePath,outputFilePath,cMin,cMax,projMatrix):
        data = np.loadtxt(inputFilePath,delimiter=",");
        label = data[:,0];
        #print tempLabel;
        matrix = data[:,range(1,data.shape[1])];
        matrix = np.divide((matrix - cMin),(cMax-cMin));
        #print tempMatrix;
        #matrix = np.loadtxt(inputFilePath,delimiter=",",usecols=range(1,35));
        #label = np.loadtxt(inputFilePath,delimiter=",",usecols=(0,));
        reducedFeatureVectors = np.dot(matrix,projMatrix);
        #Should known here that np.insert() returns a new ndarray.
        labelReducedFeatureVectors = np.insert(reducedFeatureVectors,0,label,axis=1);
        #print labelReducedFeatureVectors[0];
        np.savetxt(outputFilePath,labelReducedFeatureVectors,delimiter=",",fmt='%1.7f');
        return;
    
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
    def getNumberOfPrinciples(cls,w,threshold):
        totalEnergy = np.sum(w);
        subEnergy = 0;
        k = 0;
        while (np.divide(subEnergy,totalEnergy)<threshold):
            subEnergy = subEnergy + w[k];
            k=k+1;
            #print np.divide(subEnergy,totalEnergy);
        return k;
    
    @classmethod
    def calcF1Score(cls,testingLabels,predLabels):
        TP = 0;
        TN = 0;
        FP = 0;
        FN = 0;
        for i in range(0,len(testingLabels)):
            if(testingLabels[i]==1):
                if(predLabels[i]==testingLabels[i]):
                    TP = TP + 1;
                else:
                    FN = FN + 1; 
            else:
                if(predLabels[i]==testingLabels[i]):
                    TN = TN + 1;
                else:
                    FP = FP + 1;
        precision = 1.0*TP/(TP+FP) if (TP+FP)!=0 else 0;
        print "Precision: %f" % precision;
        recall = 1.0*TP/(TP+FN) if (TP+FN)!=0 else 0;
        print "Recall: %f" % recall;
        F1Score = 2.0*precision*recall/(precision+recall) if (precision+recall)!=0 else 0 ;
        print "F1Score: %f" % F1Score;
        return precision,recall,F1Score;
    
    @classmethod
    def splitAndSaveDatasets(cls,inputFilePath,outputFolderPath,numOfTrunks):
        data = np.loadtxt(inputFilePath,delimiter=",");
        shuffleData = np.random.permutation(data);
        
        if not os.path.exists(outputFolderPath):
            os.makedirs(outputFolderPath);
        
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