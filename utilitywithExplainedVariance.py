from pkg.dimReduction import PCAModule;
from pkg.diffPrivDimReduction import DiffPrivPCAModule;
import numpy as np;
from sklearn.model_selection import ShuffleSplit;
import matplotlib.pyplot as plt;

def drawExplainedVariance(datasetTitle,data=None,path=None,figSavedPath=None):
    plt.clf();
    if path is not None:
        data = np.loadtxt(path,delimiter=",");
    x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9];
    '''
    gaussianPercent,wishartPercent is the percentage over the non-noise PCA.
    '''
    gaussianPercent = data[:,1]/data[:,0];
    wishartPercent = data[:,2]/data[:,0];
    
    #x = [10,40,70,100,130,160,190,220,250,280,310,340];
    y1Line,y2Line = plt.plot(x, gaussianPercent, 'bo-', x, wishartPercent, 'r^-');
    if datasetTitle is 'german':
        plt.legend([y1Line,y2Line], ['Gaussian Noise','Wishart Noise'],loc=2);
    else:
        plt.legend([y1Line,y2Line], ['Gaussian Noise','Wishart Noise'],loc=4);
    
    plt.axis([0,1,0,1.1]);
    #plt.axis([0,10,0.4,1.0]);
    plt.xlabel('Epsilon',fontsize=18);
    plt.ylabel('Captured Energy',fontsize=18);
    plt.title(datasetTitle+'Dataset', fontsize=18);
    plt.xticks(x);
    if figSavedPath is None:
        plt.show();
    else:
        plt.savefig(figSavedPath+"explainedVariance_"+datasetTitle+'.pdf', format='pdf', dpi=1000);

def doExp(datasetPath,varianceRatio,numOfRounds):
    data = np.loadtxt(datasetPath,delimiter=",");
    rs = ShuffleSplit(n_splits=numOfRounds, test_size=.2, random_state=0);
    rs.get_n_splits(data);
    
    cprResult = np.zeros((9,3)); 
    
    for train_index, test_index in rs.split(data):
        
        trainingData = data[train_index];
        pureTrainingData = trainingData[:,1:];
        #trainingLabel = trainingData[:,0];
        numOfTrainingSamples = trainingData.shape[0];
        numOfFeature = trainingData.shape[1]-1;
        # Normalize data to mean 0 and standard deviation.
        #trainingColMean = np.mean(pureTrainingData,axis=0);
        #pureTrainingData = pureTrainingData - trainingColMean;
        #trainingColDeviation = np.std(pureTrainingData, axis=0);
    
        #scaledTrainingData = np.divide((pureTrainingData - trainingColMean),trainingColDeviation);
        #scaledTestingData = np.divide((pureTestingData - trainingColMean),trainingColDeviation);
    
        # Making each row unit l2 norm.
        #pureTrainingData = gf.normByRow(pureTrainingData);
        #testingData = data[test_index];
        #pureTestingData = testingData[:,1:];
        #testingLabel = testingData[:,0];
        
        pcaImpl = PCAModule.PCAImpl(pureTrainingData);
        pcaImpl.getPCs();
        numOfDimension = pcaImpl.getNumOfPCwithKPercentVariance(varianceRatio);
        print "%d/%d dimensions captures %.2f variance." % (numOfDimension,numOfFeature,varianceRatio);
        
        dpGaussianPCAImpl = DiffPrivPCAModule.DiffPrivPCAImpl(pureTrainingData);
        dpWishartPCAImpl = DiffPrivPCAModule.DiffPrivPCAImpl(pureTrainingData);
        
        pcaEnergies = pcaImpl.getEigValueEnergies();
        
        delta = np.divide(1.0,numOfTrainingSamples);
        for k in range(1,10):
            #print pcaImpl.projMatrix[:,0];
            
            epsilon = 0.1*k;
            print "epsilon: %.2f, delta: %f" % (epsilon,delta);           
            isGaussianDist = True;
            dpGaussianPCAImpl.setEpsilonAndGamma(epsilon,delta);
            dpGaussianPCAImpl.getDiffPrivPCs(isGaussianDist);
            
            isGaussianDist = False;
            dpWishartPCAImpl.setEpsilonAndGamma(epsilon,delta);
            dpWishartPCAImpl.getDiffPrivPCs(isGaussianDist);
            
            
            cprResult[k-1][0] = cprResult[k-1][0]+np.sum(pcaEnergies[:numOfDimension]);
            
            cprResult[k-1][1] = cprResult[k-1][1]+np.sum(dpGaussianPCAImpl.getEigValueEnergies()[:numOfDimension]);
            
            cprResult[k-1][2] = cprResult[k-1][2]+np.sum(dpWishartPCAImpl.getEigValueEnergies()[:numOfDimension]);
            
            print "===========================";
        
    for i in range(0,len(cprResult)):
        print "%.4f,%.4f,%.4f" % (cprResult[i][0],cprResult[i][1],cprResult[i][2]);
    print "******************************";
    
    # Compute the average value after numOfRounds experiments.
    avgCprResult = cprResult/numOfRounds;
    for i in range(0,len(cprResult)):
        print "%.3f,%.3f,%.3f" % (avgCprResult[i][0],avgCprResult[i][1],avgCprResult[i][2]);  
    
    return avgCprResult;

if __name__ == "__main__":
    
    #datasets = ['diabetes','german', 'ionosphere'];
    datasets = ['CNAE_1'];
    numOfRounds = 10;
    varianceRatio = 0.9;
    figSavedPath = "./log/";
    for dataset in datasets:
        
        print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
        datasetPath = "../distr_dp_pca/experiment/input/"+dataset+"_prePCA";
        result = doExp(datasetPath,varianceRatio,numOfRounds);
        drawExplainedVariance(dataset,data=result,figSavedPath=figSavedPath);