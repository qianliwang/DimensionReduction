from sklearn.model_selection import ShuffleSplit;
from sklearn.preprocessing import StandardScaler;
import matplotlib.pyplot as plt;

import numpy as np;
from numpy import linalg as LA;
import sys;
import os;
from multiprocessing import Pool;

from pkg.DimReduction import PCAImpl
from pkg.DPDimReduction import DiffPrivPCAImpl;
from pkg.global_functions import globalFunction as gf;

def drawVariance_x_epsilon(datasetTitle,data=None,path=None,figSavedPath=None):
    plt.clf();
    if path is not None:
        data = np.loadtxt(path, delimiter=",");
    x = np.arange(0.1, 1.1, 0.1);
    tmpDim = 1;
    for i in range(data.shape[1]):
        if data[0,i]>0.8:
            tmpDim = i;
            break;
    print "print dimension: %d" % tmpDim;
    #tmpDim = 1;
    pcaRes = [];
    gRes = [];
    wRes = [];
    pcaVal = data[np.arange(0,190,21),tmpDim];
    print pcaVal;
    for i in np.arange(0,190,21):
        tmpRange = np.arange(i+1,i+11);
        #print len(tmpRange);
        #gRes.append(data[tmpRange,tmpDim]/data[i,tmpDim]);
        gRes.append(data[tmpRange, tmpDim]);
    print gRes;
    for i in np.arange(11,200,21):
        tmpRange = np.arange(i,i+10);
        #print len(tmpRange);
        #wRes.append(data[tmpRange,tmpDim]/data[i-11,tmpDim]);
        wRes.append(data[tmpRange, tmpDim]);

    gMean, gStd = gf.calcMeanandStd(np.asarray(gRes));
    #gErrorLine = plt.errorbar(x, gMean, yerr=gStd, fmt='r', capsize=4);
    #gLine, = plt.plot(x,gMean,'r-');

    wMean,wStd = gf.calcMeanandStd(np.asarray(wRes));
    #wErrorLine = plt.errorbar(x, wMean, yerr=wStd, fmt='g', capsize=4);
    #wLine, = plt.plot(x,wMean,'g-');

    #yMin = min(np.amin(gMean),np.amin(wMean));
    #yMax = max(np.amax(gMean),np.amax(wMean));
    yMin = np.amin(gMean);
    yMax = np.amax(gMean);
    toPlot = [];
    gResArray = np.asarray(gRes);
    for i in range(gResArray.shape[1]):
        toPlot.append(gResArray[:,i]);
    ax = plt.gca();
    ax.boxplot(toPlot,widths=0.05,positions=x,showfliers=False,boxprops={'color':'indigo'});

    plt.axis([0.05, 1.05, yMin*0.94, 1.06*yMax]);
    ax.set_xticklabels(x);
    #plt.legend([gLine, wLine], ['Gaussian Noise', 'Wishart Noise'], loc=4);
    # plt.axis([0,10,0.4,1.0]);
    plt.xlabel('Epsilon', fontsize=18);
    plt.ylabel('Captured Energy', fontsize=18);
    plt.title(datasetTitle, fontsize=18);
    #plt.xticks(x);
    #plt.tight_layout();
    plt.gcf().subplots_adjust(left=0.15)
    if figSavedPath is None:
        plt.show();
    else:
        plt.savefig(figSavedPath + "explainedVariance_" + datasetTitle + '_box.pdf', format='pdf', dpi=1000);

def drawExplainedVariance(datasetTitle,data=None,path=None,figSavedPath=None):
    plt.clf();
    if path is not None:
        data = np.loadtxt(path,delimiter=",");


    '''
    x = data[:,0];
    gaussianPercent,wishartPercent is the percentage over the non-noise PCA.
    gaussianPercent = data[:,2]/data[:,1];
    wishartPercent = data[:,3]/data[:,1];
    
    y1Line,y2Line = plt.plot(x, gaussianPercent, 'bo-', x, wishartPercent, 'r^-');
    if datasetTitle is 'german':
        plt.legend([y1Line,y2Line], ['Gaussian Noise','Wishart Noise'],loc=2);
    else:
        plt.legend([y1Line,y2Line], ['Gaussian Noise','Wishart Noise'],loc=4);
    '''
    #x = range(1, data.shape[1]+1);
    if data.shape[1]<20:
        x = np.arange(1,data.shape[1]+1);
    else:
        x = np.arange(1,data.shape[1]+1,data.shape[1]/20);
    pcaIndices = np.arange(0,210,21);
    print pcaIndices;
    pcaVal = data[pcaIndices];
    pcaValMean,pcaValStd = gf.calcMeanandStd(pcaVal);
    pcaLine = plt.errorbar(x, pcaValMean[x - 1], yerr=pcaValStd[x - 1], fmt='b-', capsize=4);


    gepsiIndices = np.arange(1,210,21);
    gepsiVal = data[gepsiIndices];
    gepsiValMean,gepsiValStd = gf.calcMeanandStd(gepsiVal);
    #y1Line,y2Line = plt.plot(x, pcaValMean, 'bo-', x, pcaValStd, 'r^-');
    #gepsi1Line = plt.errorbar(x,gepsiValMean[x-1],yerr=gepsiValStd[x-1],fmt='g-',capsize=4);
    gepsiIndices = np.arange(5,210,21);
    gepsiVal = data[gepsiIndices];
    gepsiValMean,gepsiValStd = gf.calcMeanandStd(gepsiVal);
    gepsi5Line = plt.errorbar(x,gepsiValMean[x-1],yerr=gepsiValStd[x-1],fmt='r-',capsize=4);

    gepsiIndices = np.arange(9,210,21);
    gepsiVal = data[gepsiIndices];
    ggepsiValMean,gepsiValStd = gf.calcMeanandStd(gepsiVal);
    gepsi9Line = plt.errorbar(x,gepsiValMean[x-1],yerr=gepsiValStd[x-1],fmt='ro-.',capsize=4);

    wepsiIndices = np.arange(15,210,21);
    wepsiVal = data[wepsiIndices];
    wepsiValMean,wepsiValStd = gf.calcMeanandStd(wepsiVal);
    #print wepsiValStd;
    wepsi5Line = plt.errorbar(x,wepsiValMean[x-1],yerr=wepsiValStd[x-1],fmt='y-',capsize=4);
    wepsiIndices = np.arange(19,210,21);
    wepsiVal = data[wepsiIndices];
    wepsiValMean,wepsiValStd = gf.calcMeanandStd(wepsiVal);
    wepsi9Line = plt.errorbar(x,wepsiValMean[x-1],yerr=wepsiValStd[x-1],fmt='yo-',capsize=4);

    plt.axis([0,data.shape[1]+1,0,1.1]);
    #plt.axis([0,10,0.4,1.0]);
    plt.xlabel('Epsilon',fontsize=18);
    plt.ylabel('Captured Energy',fontsize=18);
    plt.title(datasetTitle, fontsize=18);
    plt.xticks(x);

    if figSavedPath is None:
        plt.show();
    else:
        plt.savefig(figSavedPath+"explainedVariance_"+datasetTitle+'.pdf', format='pdf', dpi=1000);

def calcEigRatios(eigValues):
    eigSum = np.sum(eigValues);
    tmpSum = 0;
    res = [];
    for eigVal in eigValues:
        tmpSum += eigVal;
        res.append(tmpSum/eigSum);
    return res;

def singleExp(xEpsilons,pureTrainingData,largestReducedFeature):

    numOfTrainingSamples = pureTrainingData.shape[0];
    scaler = StandardScaler(copy=False);
    # print pureTrainingData[0];
    scaler.fit(pureTrainingData);
    scaler.transform(pureTrainingData);
    # numOfFeature = trainingData.shape[1]-1;
    matrixRank = LA.matrix_rank(pureTrainingData);

    pcaImpl = PCAImpl(pureTrainingData);
    dpGaussianPCAImpl = DiffPrivPCAImpl(pureTrainingData);
    dpWishartPCAImpl = DiffPrivPCAImpl(pureTrainingData);

    pcaEnergies = pcaImpl.getEigValueEnergies();
    cprResult = [];
    cprResult.append(calcEigRatios(pcaImpl.eigValues)[:largestReducedFeature]);
    delta = np.divide(1.0, numOfTrainingSamples);
    gaussianResult = [];
    wishartResult = [];
    # print cprResult;
    for k, targetEpsilon in np.ndenumerate(xEpsilons):
        # print "epsilon: %.2f, delta: %f" % (targetEpsilon,delta);
        isGaussianDist = True;
        dpGaussianPCAImpl.setEpsilonAndGamma(targetEpsilon, delta);
        dpGaussianPCAImpl.getDiffPrivPCs(isGaussianDist, matrixRank, onlyEigvalues=True);
        # print dpGaussianPCAImpl.eigValues;
        GaussianEigRatio = calcEigRatios(dpGaussianPCAImpl.eigValues);
        gaussianResult.append(GaussianEigRatio[:largestReducedFeature]);
        # print GaussianEigRatio;
        isGaussianDist = False;
        dpWishartPCAImpl.setEpsilonAndGamma(targetEpsilon, delta);
        dpWishartPCAImpl.getDiffPrivPCs(isGaussianDist, matrixRank, onlyEigvalues=True);
        WishartEigRatio = calcEigRatios(dpWishartPCAImpl.eigValues);
        wishartResult.append(WishartEigRatio[:largestReducedFeature]);
        # print WishartEigRatio;
    cprResult.extend(gaussianResult);
    cprResult.extend(wishartResult);
    # print cprResult;
    return np.asarray(cprResult);

def doExp(datasetPath,varianceRatio,numOfRounds):
    if os.path.basename(datasetPath).endswith('npy'):
        data = np.load(datasetPath);
    else:
        data = np.loadtxt(datasetPath, delimiter=",");

    rs = ShuffleSplit(n_splits=numOfRounds, test_size=2, random_state=0);
    rs.get_n_splits(data);
    globalPCA = PCAImpl(data[:, 1:]);
    numOfFeature = data.shape[1] - 1;
    matrixRank = LA.matrix_rank(data[:, 1:]);

    print "Matrix rank of the data is %d." % matrixRank;
    largestReducedFeature = globalPCA.getNumOfPCwithKPercentVariance(varianceRatio);
    print "%d/%d dimensions captures %.2f variance." % (largestReducedFeature, numOfFeature, varianceRatio);

    xEpsilons = np.arange(0.1, 1.1, 0.1);
    # print xDimensions;
    # p = Pool(numOfRounds);
    # allResults = [];
    cprResult = [];
    m = 0;
    for train_index, test_index in rs.split(data):
        print "Trail %d" % m;
        trainingData = data[train_index];
        pureTrainingData = trainingData[:, 1:];
        tmpResult = singleExp(xEpsilons, pureTrainingData, largestReducedFeature);
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
    for result in cprResult:
        print ','.join(['%.3f' % num for num in result]);

    return np.asarray(cprResult, dtype=float);

if __name__ == "__main__":
    #datasets = ['diabetes','german', 'ionosphere'];
    numOfRounds = 10;
    varianceRatio = 0.9;
    figSavedPath = "./fig/";
    resultSavedPath = "./log/firstRevision/";
    if len(sys.argv) >1:
        datasetPath = sys.argv[1];
        print "+++ using passed in arguments: %s" % (datasetPath);
        result = doExp(datasetPath,varianceRatio,numOfRounds);
        np.savetxt(resultSavedPath+"explainedVariance_"+os.path.basename(datasetPath)+".output",result,delimiter=",",fmt='%1.3f');
    else:
        datasets = ['Million Song','Aloi','Facebook','Amazon','YaleB','p53 Mutant','MovieLens','CNAE_3','CNAE_2','CNAE_5','CNAE_7','Amazon_3','madelon'];
        for dataset in datasets:  
            print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
            datasetPath = "./input/"+dataset+"_prePCA";
            #result = doExp(datasetPath,varianceRatio,numOfRounds);
            #np.savetxt(resultSavedPath+"explainedVariance_"+dataset+".output",result,delimiter=",",fmt='%1.3f');
            #drawExplainedVariance(dataset,data=None,path=resultSavedPath+"explainedVariance_"+dataset+".output",figSavedPath=None);
            drawVariance_x_epsilon(dataset,data=None,path=resultSavedPath+"explainedVariance_"+dataset+".output",figSavedPath=resultSavedPath);