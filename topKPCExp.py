from pkg.dimReduction import PCAModule;
from pkg.diffPrivDimReduction import DiffPrivPCAModule;
import numpy as np;
from numpy import linalg as LA;
from sklearn.model_selection import ShuffleSplit;
import sys;
import os;
from multiprocessing import Pool;
from pkg.global_functions import globalFunction as gf;
from pkg.diffPrivDimReduction.DPModule import DiffPrivImpl;
import scipy.sparse as sparse;
from sklearn.preprocessing import StandardScaler;
import re;

def singleExp(pureTrainingData,targetEpsilon):

    scaler = StandardScaler(copy=False);
    # print pureTrainingData[0];
    scaler.fit(pureTrainingData);
    scaler.transform(pureTrainingData);

    pcaImpl = PCAModule.PCAImpl(pureTrainingData);
    print "First Principal Component: %f" % pcaImpl.getEigValueEnergies()[0];
    pcaImpl.genEigenvectors_power(pcaImpl.covMatrix,1);

    dpGaussianPCAImpl = DiffPrivPCAModule.DiffPrivPCAImpl(pureTrainingData);
    dpWishartPCAImpl = DiffPrivPCAModule.DiffPrivPCAImpl(pureTrainingData);
    #print dpGaussianPCAImpl.frobeniusSensitivity;
    # dpGaussianPCAImpl.L2Sensitivity = 1;
    # dpWishartPCAImpl.L2Sensitivity = 1;
    #print dpWishartPCAImpl.L2Sensitivity;

    cprResult = [];
    gaussianResult = [];
    wishartResult = [];
    delta = np.divide(1.0,pureTrainingData.shape[0]);

    gaussianNoise = DiffPrivImpl.SymmGaussian(targetEpsilon,delta,dpGaussianPCAImpl.covMatrix.shape[0],dpGaussianPCAImpl.frobeniusSensitivity);
    #print LA.svd(gaussianNoise,compute_uv=False);
    noisyCov = dpGaussianPCAImpl.covMatrix + gaussianNoise;
    dpGaussianPCAImpl.genEigenvectors_power(noisyCov,1);

    wishartNoise = DiffPrivImpl.SymmWishart_withDelta(targetEpsilon, delta, dpWishartPCAImpl.covMatrix.shape[0], dpWishartPCAImpl.frobeniusSensitivity);
    #print LA.svd(wishartNoise, compute_uv=False);
    noisyCov = dpGaussianPCAImpl.covMatrix + wishartNoise;
    dpWishartPCAImpl.genEigenvectors_power(noisyCov,1);

    cprResult.extend(gaussianResult);
    cprResult.extend(wishartResult);
    return np.asarray(cprResult);


def doExp(datasetPath, targetEpsilon, numOfRounds):
    if os.path.basename(datasetPath).endswith('npy'):
        data = np.load(datasetPath);
    else:
        data = np.loadtxt(datasetPath, delimiter=",");
    rs = ShuffleSplit(n_splits=numOfRounds, test_size=1, random_state=0);
    rs.get_n_splits(data);
    print "Samples: %d, Features: %d" % (data.shape[0],data.shape[1]-1);

    # p = Pool(numOfRounds);
    cprResult = [];
    m = 0;
    for train_index, test_index in rs.split(data):
        print "Trail %d" % m;
        trainingData = data[train_index,1:];
        tmpResult = singleExp(trainingData,targetEpsilon);
        cprResult.extend(tmpResult);
        m += 1;
        # tmpResult = p.apply_async(singleExp, (xEpsilons,pureTrainingData,largestReducedFeature));
        # cprResult += tmpResult.get();

    # Compute the average value after numOfRounds experiments.
    # p.close();
    # p.join();

    return cprResult;
def processOutput(path):
    rawResult = [];
    with open(path, 'r') as f:
        for content in f:
            m = re.search('\[(\d+)\]', content);
            if m is not None:
                #print content;
                #print m.group(1);
                rawResult.append(int(m.group(1)));
    #print rawResult;
    rawArray = np.asarray(rawResult);
    rawArray = np.reshape(rawArray,(-1,3));
    #print rawArray;
    xTrails = np.arange(0,rawArray.shape[0],10);
    for i in xTrails:
        tmpMean,tmpStd = gf.calcMeanandStd(rawArray[i:i+10]);
        print tmpMean,tmpStd;

if __name__ == "__main__":
    # datasets = ['diabetes','german', 'ionosphere'];
    numOfRounds = 10;
    targetEpsilon = 0.9;
    figSavedPath = "./fig/";
    resultSavedPath = "./log/";
    if len(sys.argv) > 1:
        datasetPath = sys.argv[1];
        print "+++ using passed in arguments: %s" % (datasetPath);
        result = doExp(datasetPath,targetEpsilon, numOfRounds);
        np.savetxt(resultSavedPath + "explainedVariance_" + os.path.basename(datasetPath) + ".output", result,
                   delimiter=",", fmt='%1.3f');
    else:
        datasets = ['aloi','facebook','Music'];
        for dataset in datasets:
            print "++++++++++++++++++++++++++++  " + dataset + "  +++++++++++++++++++++++++";
            datasetPath = "./input/" + dataset + "_prePCA";
            #result = doExp(datasetPath,targetEpsilon,numOfRounds);
            #np.savetxt(resultSavedPath+"explainedVariance_"+dataset+".output",result,delimiter=",",fmt='%1.3f');
        outputPath = "./log/topk.output";
        processOutput(outputPath);