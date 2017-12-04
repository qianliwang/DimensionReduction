from pkg.dimReduction import PCAModule;
from pkg.diffPrivDimReduction import DiffPrivPCAModule;
import numpy as np;
from sklearn.model_selection import ShuffleSplit;
import matplotlib.pyplot as plt;
import sys;
import os;

def drawExplainedVariance(datasetTitle,data=None,path=None,figSavedPath=None):
    plt.clf();
    if path is not None:
        data = np.loadtxt(path,delimiter=",");
    x = data[:,0];
    '''
    gaussianPercent,wishartPercent is the percentage over the non-noise PCA.
    '''
    gaussianPercent = data[:,2]/data[:,1];
    wishartPercent = data[:,3]/data[:,1];
    
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
    
    xEpsilons = np.arange(0.1,1.0,0.1);
    cprResult = np.zeros((len(xEpsilons),4));
    #print xDimensions;
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
        #for k in range(1,10):
        for k, targetEpsilon in np.ndenumerate(xEpsilons):
            #print pcaImpl.projMatrix[:,0];
            
            print "epsilon: %.2f, delta: %f" % (targetEpsilon,delta);           
            isGaussianDist = True;
            dpGaussianPCAImpl.setEpsilonAndGamma(targetEpsilon,delta);
            dpGaussianPCAImpl.getDiffPrivPCs(isGaussianDist);
            
            isGaussianDist = False;
            dpWishartPCAImpl.setEpsilonAndGamma(targetEpsilon,delta);
            dpWishartPCAImpl.getDiffPrivPCs(isGaussianDist);
            
            cprResult[k][0] = cprResult[k][0]+targetEpsilon;
            cprResult[k][1] = cprResult[k][1]+np.sum(pcaEnergies[:numOfDimension]);
            cprResult[k][2] = cprResult[k][2]+np.sum(dpGaussianPCAImpl.getEigValueEnergies()[:numOfDimension]);
            cprResult[k][3] = cprResult[k][3]+np.sum(dpWishartPCAImpl.getEigValueEnergies()[:numOfDimension]);
            
            print "===========================";
    """    
    for i in range(0,len(cprResult)):
        print "%.4f,%.4f,%.4f" % (cprResult[i][0],cprResult[i][1],cprResult[i][2]);
    print "******************************";
    """
    # Compute the average value after numOfRounds experiments.
    avgCprResult = cprResult/numOfRounds;
    for result in avgCprResult:
        print "%.2f,%.3f,%.3f,%.3f" % (result[0],result[1],result[2],result[3]);  
    
    return avgCprResult;

if __name__ == "__main__":
    
    #datasets = ['diabetes','german', 'ionosphere'];
    numOfRounds = 4;
    varianceRatio = 0.9;
    figSavedPath = "./log/";
    resultSavedPath = "./log/";
    if len(sys.argv) >1:
        datasetPath = sys.argv[1];
        print "+++ using passed in arguments: %s" % (datasetPath);
        result = doExp(datasetPath,varianceRatio,numOfRounds);
        np.savetxt(resultSavedPath+"explainedVariance_"+os.path.basename(datasetPath)+".output",result,delimiter=",");
    else:
        datasets = ['Amazon_3','face2','madelon','CNAE_2'];
        for dataset in datasets:    
            print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
            datasetPath = "./input/"+dataset+"_prePCA";
            result = doExp(datasetPath,varianceRatio,numOfRounds);
            np.savetxt(resultSavedPath+"explainedVariance_"+dataset+".output",result,delimiter=",");
            #drawExplainedVariance(dataset,data=result,figSavedPath=figSavedPath);