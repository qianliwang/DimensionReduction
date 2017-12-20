from pkg.dimReduction import PCAModule;
from pkg.diffPrivDimReduction import DiffPrivPCAModule;
import numpy as np;
from numpy import linalg as LA;
from sklearn.model_selection import ShuffleSplit;
import matplotlib.pyplot as plt;
import sys;
import os;
from multiprocessing import Pool;
from pkg.global_functions import globalFunction as gf;
from pkg.diffPrivDimReduction.DPModule import DiffPrivImpl;
import scipy.sparse as sparse;

def singleExp(pureTrainingData):

    pcaImpl = PCAModule.PCAImpl(pureTrainingData);

    pcaImpl.genEigenvectors_power(pcaImpl.covMatrix,1);

    dpGaussianPCAImpl = DiffPrivPCAModule.DiffPrivPCAImpl(pureTrainingData);
    dpWishartPCAImpl = DiffPrivPCAModule.DiffPrivPCAImpl(pureTrainingData);
    print dpGaussianPCAImpl.frobeniusSensitivity;
    # dpGaussianPCAImpl.L2Sensitivity = 1;
    # dpWishartPCAImpl.L2Sensitivity = 1;
    print dpWishartPCAImpl.L2Sensitivity;

    cprResult = [];
    gaussianResult = [];
    wishartResult = [];
    targetEpsilon = 0.5;
    delta = np.divide(1.0,pureTrainingData.shape[0]);
    # print "epsilon: %.2f, delta: %f" % (targetEpsilon,delta);

    gaussianNoise = DiffPrivImpl.SymmGaussian(targetEpsilon,delta,dpGaussianPCAImpl.covMatrix.shape[0],dpGaussianPCAImpl.frobeniusSensitivity);
    noisyCov = dpGaussianPCAImpl.covMatrix + gaussianNoise;
    dpGaussianPCAImpl.genEigenvectors_power(noisyCov,1);
    # print dpGaussianPCAImpl.eigValues;
    # print GaussianEigRatio;
    wishartNoise = DiffPrivImpl.SymmWishart_withDelta(targetEpsilon, delta, dpWishartPCAImpl.covMatrix.shape[0], dpWishartPCAImpl.frobeniusSensitivity);
    noisyCov = dpGaussianPCAImpl.covMatrix + wishartNoise;
    dpWishartPCAImpl.genEigenvectors_power(noisyCov,1);

    # print WishartEigRatio;
    cprResult.extend(gaussianResult);
    cprResult.extend(wishartResult);
    # print cprResult;
    return np.asarray(cprResult);


def doExp(datasetPath, varianceRatio, numOfRounds):
    data = np.loadtxt(datasetPath, delimiter=",");
    rs = ShuffleSplit(n_splits=numOfRounds, test_size=.2, random_state=0);
    rs.get_n_splits(data);


    # print xDimensions;
    # p = Pool(numOfRounds);
    # allResults = [];
    cprResult = [];
    m = 0;
    for train_index, test_index in rs.split(data):
        print "Trail %d" % m;
        trainingData = data[train_index];
        tmpResult = singleExp(trainingData);
        cprResult.extend(tmpResult);
        m += 1;
        # print tmpResult.shape;
        # print tmpResult;
        # tmpResult = p.apply_async(singleExp, (xEpsilons,pureTrainingData,largestReducedFeature));
        # cprResult += tmpResult.get();
    """
        for i in range(0,len(cprResult)):
            print "%.4f,%.4f,%.4f" % (cprResult[i][0],cprResult[i][1],cprResult[i][2]);
        print "******************************";
    """
    # Compute the average value after numOfRounds experiments.
    # avgCprResult = cprResult/numOfRounds;
    # p.close();
    # p.join();
    """
    for result in cprResult:
        print "%.2f,%.3f,%.3f,%.3f" % (result[0],result[1],result[2],result[3]);  

    """

    return cprResult;


if __name__ == "__main__":
    # datasets = ['diabetes','german', 'ionosphere'];
    numOfRounds = 5;
    varianceRatio = 0.9;
    figSavedPath = "./fig/";
    resultSavedPath = "./log/";
    if len(sys.argv) > 1:
        datasetPath = sys.argv[1];
        print "+++ using passed in arguments: %s" % (datasetPath);
        result = doExp(datasetPath, varianceRatio, numOfRounds);
        np.savetxt(resultSavedPath + "explainedVariance_" + os.path.basename(datasetPath) + ".output", result,
                   delimiter=",", fmt='%1.3f');
    else:
        datasets = ['CNAE_2', 'Face_15', 'diabetes', 'german', 'Amazon', 'p53 Mutant', 'diabetes', 'MovieLens',
                    'ionosphere', 'CNAE_3', 'CNAE_2', 'CNAE_5', 'CNAE_7', 'Amazon_3', 'madelon'];
        for dataset in datasets:
            print "++++++++++++++++++++++++++++  " + dataset + "  +++++++++++++++++++++++++";
            datasetPath = "./input/" + dataset + "_prePCA";
            result = doExp(datasetPath,varianceRatio,numOfRounds);
            #np.savetxt(resultSavedPath+"explainedVariance_"+dataset+".output",result,delimiter=",",fmt='%1.3f');
