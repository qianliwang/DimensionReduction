from pkg.dimReduction import PCAModule;
from pkg.diffPrivDimReduction import DiffPrivPCAModule;
import numpy as np;
from numpy import linalg as LA;
from sklearn.model_selection import ShuffleSplit;
import matplotlib.pyplot as plt;
import sys;
import os;
from multiprocessing import Pool;

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

def singleExp(xEpsilons,pureTrainingData,largestReducedFeature):
    
    cprResult = np.zeros((len(xEpsilons),4));
    numOfTrainingSamples = pureTrainingData.shape[0];
    #numOfFeature = trainingData.shape[1]-1;
    matrixRank = LA.matrix_rank(pureTrainingData);
    pcaImpl = PCAModule.PCAImpl(pureTrainingData);
    pcaImpl.getPCs(matrixRank);
    
    dpGaussianPCAImpl = DiffPrivPCAModule.DiffPrivPCAImpl(pureTrainingData);
    dpWishartPCAImpl = DiffPrivPCAModule.DiffPrivPCAImpl(pureTrainingData);
    
    pcaEnergies = pcaImpl.getEigValueEnergies();
    
    delta = np.divide(1.0,numOfTrainingSamples);
    
    for k, targetEpsilon in np.ndenumerate(xEpsilons):
        #print pcaImpl.projMatrix[:,0];
        #print "epsilon: %.2f, delta: %f" % (targetEpsilon,delta);           
        isGaussianDist = True;
        dpGaussianPCAImpl.setEpsilonAndGamma(targetEpsilon,delta);
        dpGaussianPCAImpl.getDiffPrivPCs(isGaussianDist,matrixRank);
        
        isGaussianDist = False;
        dpWishartPCAImpl.setEpsilonAndGamma(targetEpsilon,delta);
        dpWishartPCAImpl.getDiffPrivPCs(isGaussianDist,matrixRank);
        
        cprResult[k][0] = cprResult[k][0]+targetEpsilon;
        cprResult[k][1] = cprResult[k][1]+np.sum(pcaEnergies[:largestReducedFeature]);
        cprResult[k][2] = cprResult[k][2]+np.sum(dpGaussianPCAImpl.getEigValueEnergies()[:largestReducedFeature]);
        cprResult[k][3] = cprResult[k][3]+np.sum(dpWishartPCAImpl.getEigValueEnergies()[:largestReducedFeature]);
    return cprResult;

def doExp(datasetPath,varianceRatio,numOfRounds):
    data = np.loadtxt(datasetPath,delimiter=",");
    rs = ShuffleSplit(n_splits=numOfRounds, test_size=.2, random_state=0);
    rs.get_n_splits(data);
    globalPCA = PCAModule.PCAImpl(data[:,1:]);
    numOfFeature = data.shape[1]-1;
    matrixRank = LA.matrix_rank(data[:,1:]);

    print "Matrix rank of the data is %d." % matrixRank;
    largestReducedFeature = globalPCA.getNumOfPCwithKPercentVariance(varianceRatio);
    print "%d/%d dimensions captures %.2f variance." % (largestReducedFeature,numOfFeature,varianceRatio);
    
    xEpsilons = np.arange(0.1,1.0,0.1);
    cprResult = np.zeros((len(xEpsilons),4));
    #print xDimensions;
    p = Pool(numOfRounds);
    #allResults = [];
    for train_index, test_index in rs.split(data):
        trainingData = data[train_index];
        pureTrainingData = trainingData[:,1:];
        tmpResult = p.apply_async(singleExp, (xEpsilons,pureTrainingData,largestReducedFeature));
        cprResult += tmpResult.get();
    """
        for i in range(0,len(cprResult)):
            print "%.4f,%.4f,%.4f" % (cprResult[i][0],cprResult[i][1],cprResult[i][2]);
        print "******************************";
    """
    # Compute the average value after numOfRounds experiments.
    avgCprResult = cprResult/numOfRounds;
    p.close();
    p.join();
    for result in avgCprResult:
        print "%.2f,%.3f,%.3f,%.3f" % (result[0],result[1],result[2],result[3]);  
    
    return avgCprResult;

if __name__ == "__main__":
    #datasets = ['diabetes','german', 'ionosphere'];
    numOfRounds = 10;
    varianceRatio = 0.9;
    figSavedPath = "./log/";
    resultSavedPath = "./log/";
    if len(sys.argv) >1:
        datasetPath = sys.argv[1];
        print "+++ using passed in arguments: %s" % (datasetPath);
        result = doExp(datasetPath,varianceRatio,numOfRounds);
        np.savetxt(resultSavedPath+"explainedVariance_"+os.path.basename(datasetPath)+".output",result,delimiter=",",fmt='%1.3f');
    else:
        datasets = ['ionosphere','CNAE_2','CNAE_5','CNAE_7','face2','Amazon_3','madelon'];
        for dataset in datasets:  
            print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
            datasetPath = "./input/"+dataset+"_prePCA";
            result = doExp(datasetPath,varianceRatio,numOfRounds);
            np.savetxt(resultSavedPath+"explainedVariance_"+dataset+".output",result,delimiter=",",fmt='%1.3f');
            #drawExplainedVariance(dataset,data=result,figSavedPath=figSavedPath);