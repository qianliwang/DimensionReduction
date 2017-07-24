from pkg.global_functions import globalFunction as gf;
from pkg.svm import SVMModule;
from pkg.dimReduction import PCAModule;
from pkg.diffPrivDimReduction import DiffPrivPCAModule;
import numpy as np;

if __name__ == "__main__":
    
    datasets = ['australian','diabetes','german', 'ionosphere'];
    
    for dataset in datasets:
        
        print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
        datasetPath = "../distr_dp_pca/experiment/input/"+dataset+"_prePCA";
        trainingDataPath = datasetPath+"_training";
        testingDataPath = datasetPath+"_testing";
        #for i in range(10):
        
        totalRound = 10;
        
        gf.genTrainingTestingData(datasetPath,trainingDataPath,testingDataPath);
        
        trainingData = np.loadtxt(trainingDataPath,delimiter=",");
        pureTrainingData = trainingData[:,1:];
        trainingLabel = trainingData[:,0];
        
        numOfDimension = int(trainingData.shape[1]/2.0);
        #trainingColMean = np.mean(pureTrainingData,axis=0);
        #trainingColDeviation = np.std(pureTrainingData, axis=0);
        
        #scaledTrainingData = np.divide((pureTrainingData - trainingColMean),trainingColDeviation);
        #scaledTestingData = np.divide((pureTestingData - trainingColMean),trainingColDeviation);
        
        testingData = np.loadtxt(testingDataPath,delimiter=",");
        pureTestingData = testingData[:,1:];
        testingLabel = testingData[:,0];
        
        pcaImpl = PCAModule.PCAImpl(pureTrainingData);
        pcaImpl.getPCs();
        
        dpGaussianPCAImpl = DiffPrivPCAModule.DiffPrivPCAImpl(pureTrainingData);
        dpWishartPCAImpl = DiffPrivPCAModule.DiffPrivPCAImpl(pureTrainingData);
              
        cprResult = np.zeros((9,3));       
        
        for k in range(1,10):
            #print pcaImpl.projMatrix[:,0];
            
            epsilon = 0.1;
            delta = 0.01;
            epsilon = 0.1*k;
            
            for j in range(totalRound):
                    
                isGaussianDist = True;
                dpGaussianPCAImpl.setEpsilonAndGamma(epsilon,delta);
                dpGaussianPCAImpl.getDiffPrivPCs(isGaussianDist);
                
                isGaussianDist = False;
                dpWishartPCAImpl.setEpsilonAndGamma(epsilon,delta);
                dpWishartPCAImpl.getDiffPrivPCs(isGaussianDist);
                
                projTrainingData1 = np.dot(pureTrainingData,pcaImpl.projMatrix[:,0:numOfDimension]);
                projTestingData1 = np.dot(pureTestingData,pcaImpl.projMatrix[:,0:numOfDimension]);
                #print projTrainingData.shape;
                
                result = SVMModule.SVMClf.rbfSVM(projTrainingData1,trainingLabel,projTestingData1,testingLabel);
                cprResult[k-1][0] = cprResult[k-1][0]+result[2];
                
                
                projTrainingData2 = np.dot(pureTrainingData,dpGaussianPCAImpl.projMatrix[:,0:numOfDimension]);
                projTestingData2 = np.dot(pureTestingData,dpGaussianPCAImpl.projMatrix[:,0:numOfDimension]);
                #print projTestingData.shape;
                result = SVMModule.SVMClf.rbfSVM(projTrainingData2,trainingLabel,projTestingData2,testingLabel);
                cprResult[k-1][1] = cprResult[k-1][1]+result[2];
                
                projTrainingData3 = np.dot(pureTrainingData,dpWishartPCAImpl.projMatrix[:,0:numOfDimension]);
                projTestingData3 = np.dot(pureTestingData,dpWishartPCAImpl.projMatrix[:,0:numOfDimension]);
                #print projTestingData.shape;
                result = SVMModule.SVMClf.rbfSVM(projTrainingData3,trainingLabel,projTestingData3,testingLabel);
                cprResult[k-1][2] = cprResult[k-1][2]+result[2];
                
                print "===========================";
        
            for i in range(0,len(cprResult)):
                print "%f , %f, %f" % (cprResult[i][0],cprResult[i][1],cprResult[i][2]);
        
        print "******************************";
        for i in range(0,len(cprResult)):
            print "%f , %f, %f " % (cprResult[i][0]/totalRound,cprResult[i][1]/totalRound,cprResult[i][2]/totalRound);
            