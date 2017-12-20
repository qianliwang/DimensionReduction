from pkg.dimReduction import PCAModule;
from pkg.diffPrivDimReduction import DiffPrivPCAModule;
import numpy as np;
from numpy import linalg as LA;
from sklearn.model_selection import ShuffleSplit;
import sys;
import os;
from multiprocessing import Pool;
from sklearn.preprocessing import StandardScaler;

def calcEigRatios(eigValues):
    eigSum = np.sum(eigValues);
    tmpSum = 0;
    res = [];
    for eigVal in eigValues:
        tmpSum += eigVal;
        res.append(tmpSum/eigSum);
    return res;
def singleExp(xEpsilons,pureTrainingData,largestReducedFeature):
    
    #cprResult = np.zeros((len(xEpsilons),4));
    numOfTrainingSamples = pureTrainingData.shape[0];
    scaler = StandardScaler(copy=False);
    # print pureTrainingData[0];
    scaler.fit(pureTrainingData);
    scaler.transform(pureTrainingData);
    #numOfFeature = trainingData.shape[1]-1;
    matrixRank = LA.matrix_rank(pureTrainingData);

    pcaImpl = PCAModule.PCAImpl(pureTrainingData);
    dpGaussianPCAImpl = DiffPrivPCAModule.DiffPrivPCAImpl(pureTrainingData);
    dpWishartPCAImpl = DiffPrivPCAModule.DiffPrivPCAImpl(pureTrainingData);
    
    pcaEnergies = pcaImpl.getEigValueEnergies();
    cprResult = [];
    cprResult.append(calcEigRatios(pcaImpl.eigValues)[:largestReducedFeature]);
    delta = np.divide(1.0,numOfTrainingSamples);
    gaussianResult = [];
    wishartResult = [];
    #print cprResult;
    for k, targetEpsilon in np.ndenumerate(xEpsilons):
        #print "epsilon: %.2f, delta: %f" % (targetEpsilon,delta);
        isGaussianDist = True;
        dpGaussianPCAImpl.setEpsilonAndGamma(targetEpsilon,delta);
        dpGaussianPCAImpl.getDiffPrivPCs(isGaussianDist,matrixRank,onlyEigvalues=True);
        #print dpGaussianPCAImpl.eigValues;
        GaussianEigRatio = calcEigRatios(dpGaussianPCAImpl.eigValues);
        gaussianResult.append(GaussianEigRatio[:largestReducedFeature]);
        #print GaussianEigRatio;
        isGaussianDist = False;
        dpWishartPCAImpl.setEpsilonAndGamma(targetEpsilon,delta);
        dpWishartPCAImpl.getDiffPrivPCs(isGaussianDist,matrixRank,onlyEigvalues=True);
        WishartEigRatio = calcEigRatios(dpWishartPCAImpl.eigValues);
        wishartResult.append(WishartEigRatio[:largestReducedFeature]);
        #print WishartEigRatio;
    cprResult.extend(gaussianResult);
    cprResult.extend(wishartResult);
    #print cprResult;
    return np.asarray(cprResult);

def doExp(datasetPath,varianceRatio,numOfRounds):

    if os.path.basename(datasetPath).endswith('npy'):
        data = np.load(datasetPath);
    else:
        data = np.loadtxt(datasetPath, delimiter=",");

    rs = ShuffleSplit(n_splits=numOfRounds, test_size=.2, random_state=0);
    rs.get_n_splits(data);
    globalPCA = PCAModule.PCAImpl(data[:,1:]);
    numOfFeature = data.shape[1]-1;
    matrixRank = LA.matrix_rank(data[:,1:]);

    print "Matrix rank of the data is %d." % matrixRank;
    largestReducedFeature = globalPCA.getNumOfPCwithKPercentVariance(varianceRatio);
    print "%d/%d dimensions captures %.2f variance." % (largestReducedFeature,numOfFeature,varianceRatio);
    
    xEpsilons = np.arange(0.1,1.1,0.1);
    cprResult = np.zeros((len(xEpsilons),4));
    #print xDimensions;
    #p = Pool(numOfRounds);
    #allResults = [];
    cprResult = [];
    m =0;
    for train_index, test_index in rs.split(data):
        print "Trail %d" % m;
        trainingData = data[train_index];
        pureTrainingData = trainingData[:,1:];
        tmpResult = singleExp(xEpsilons,pureTrainingData,largestReducedFeature);
        cprResult.extend(tmpResult);
        m += 1;
        #print tmpResult.shape;
        #print tmpResult;
        #tmpResult = p.apply_async(singleExp, (xEpsilons,pureTrainingData,largestReducedFeature));
        #cprResult += tmpResult.get();
    """
        for i in range(0,len(cprResult)):
            print "%.4f,%.4f,%.4f" % (cprResult[i][0],cprResult[i][1],cprResult[i][2]);
        print "******************************";
    """
    # Compute the average value after numOfRounds experiments.
    #avgCprResult = cprResult/numOfRounds;
    #p.close();
    #p.join();
    """
    for result in cprResult:
        print "%.2f,%.3f,%.3f,%.3f" % (result[0],result[1],result[2],result[3]);  
    
    """

    return np.asarray(cprResult,dtype=float);

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
        datasets = ['CNAE_3'];
        for dataset in datasets:  
            print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
            datasetPath = "./input/"+dataset+"_prePCA";
            result = doExp(datasetPath,varianceRatio,numOfRounds);
            np.savetxt(resultSavedPath+"explainedVariance_"+dataset+".output",result,delimiter=",",fmt='%1.3f');
