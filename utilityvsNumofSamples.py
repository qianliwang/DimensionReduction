from sklearn.model_selection import ShuffleSplit;
from sklearn.model_selection import StratifiedShuffleSplit;
from sklearn.preprocessing import StandardScaler;
from sklearn import preprocessing;

import numpy as np;
from numpy import linalg as LA;
from numpy.linalg import norm;
import scipy.sparse as sparse;
import sys;
import os;
from multiprocessing import Pool;

from pkg import SVMModule;
from pkg.DimReduction import PCAImpl;
from pkg import invwishart;
from pkg.global_functions import globalFunction as gf;
from pkg.DPDimReduction import DiffPrivImpl;
from pkg.global_functions import globalFunction as gf;



def drawScore(datasetTitle, data=None, path=None,n_trails=1, type='Accuracy',figSavedPath=None):
    import matplotlib;
    import matplotlib.pyplot as plt;
    from matplotlib.ticker import FuncFormatter
    def to_percent(y, position):
        # Display the ylabel in percent.
        # Ignore the passed in position. This has the effect of scaling the default
        # tick locations.
        s = str(100 * y)

        # The percent symbol needs escaping in latex
        if matplotlib.rcParams['text.usetex'] is True:
            return s + r'$\%$'
        else:
            return s + '%'

    plt.clf();
    if path is not None:
        data = np.loadtxt(path, delimiter=",");
    n_dim = data.shape[0];
    if n_trails is not 1:
        n_dim = int(data.shape[0]/n_trails);
        data = data.reshape(n_trails,-1,data.shape[1]);
        data_mean,data_std = gf.calcMeanandStd(data);
    else:
        data_mean = data;
        data_std = np.zeros(data.shape);
    print "Number of points on x-axis: %d" % n_dim;
    x = data_mean[:, 0];

    if type is 'f1':
        #drawF1Score(datasetTitle, data=data, path=path, figSavedPath=figSavedPath);
        data_mean = data_mean[:,[3,7,11]];
        data_std = data_std[:,[3,7,11]];
    elif type is 'accuracy':
        #drawAccuracy(datasetTitle, data=data, path=path, figSavedPath=figSavedPath);
        data_mean = data_mean[:, [4, 8, 12]];
        data_std = data_std[:, [4, 8, 12]];
    """
    minVector = np.amin(data[:,1:],axis=0);
    yMin = min(minVector);
    maxVector = np.amax(data[:,1:],axis=0);
    yMax = max(maxVector);

    yMin = (yMin-0.1) if (yMin-0.1)>0 else 0;
    yMax = (yMax+0.1) if (yMax+0.1)<1 else 1;
    #x = [10,40,70,100,130,160,190,220,250,280,310,340];
    y1Line,y2Line,y3Line = plt.plot(x, data[:,1], 'bo-', x, data[:,2], 'r^-',x, data[:,3], 'gs-');
    if datasetTitle is 'Ionosphere':
        plt.legend([y1Line,y2Line,y3Line], ['PCA','DPDPCA','PrivateLocalPCA'],loc=4);
    else:
        plt.legend([y1Line,y2Line,y3Line], ['PCA','DPDPCA','PrivateLocalPCA'],loc=2);
    """
    #x = np.arange(10,60,10);
    pcaF1 = [];
    dpdpcaF1 = [];
    privateF1 = [];
    '''
    for i in range(0, len(x)):
        pcaIndices = np.arange(i, data.shape[0], len(x));
        print pcaIndices;
        pcaF1.append(data[pcaIndices, 3]);
        dpdpcaF1.append(data[pcaIndices, 7]);
        privateF1.append(data[pcaIndices, 11]);
    # print np.asarray(gF1);
    '''
    ax = plt.gca();
    width = 2.3;

    #pcaF1Mean, pcaF1Std = gf.calcMeanandStd(np.asarray(pcaF1).T);
    #pcaF1ErrorLine = plt.errorbar(x, pcaF1Mean, yerr=pcaF1Std, fmt='b', capsize=4);
    #pcaF1Line, = plt.plot(x,pcaF1Mean,'b-');
    #dpdpcaF1Mean, dpdpcaF1Std = gf.calcMeanandStd(np.asarray(dpdpcaF1).T);
    #dpdpcaF1ErrorLine = plt.errorbar(x, dpdpcaF1Mean, yerr=dpdpcaF1Std, fmt='m', capsize=4);
    #dpdpcaF1Line, = plt.plot(x,dpdpcaF1Mean,'m-');
    #privateF1Mean, privateF1Std = gf.calcMeanandStd(np.asarray(privateF1).T);
    #privateF1ErrorLine = plt.errorbar(x, privateF1Mean, yerr=privateF1Std, fmt='c', capsize=4);
    #privateF1Line, = plt.plot(x,privateF1Mean,'c-');

    pcaBar = ax.bar(x + 2.5, data_mean[:,0], width, color='b', yerr=data_std[:,0], capsize=2);
    dpdpcaBar = ax.bar(x - 2.5, data_mean[:,1], width, color='m', yerr=data_std[:,1], capsize=2);
    privateBar = ax.bar(x, data_mean[:,2], width, color='c', yerr=data_std[:,2], capsize=2);

    yMin = round(np.amin(data_mean[:,2]),1)-0.1;
    if datasetTitle=='GISETTE':
        plt.axis([5, 55, yMin, 1.05]);
    else:
        plt.axis([5, 55, yMin, 1.13]);
    """
    if 'p53' in datasetTitle:
        plt.legend([pcaF1Line, dpdpcaF1Line, privateF1Line], ['PCA', 'DPDPCA', 'PrivateLocalPCA'], loc=2, fontsize='small');
    else:
        plt.legend([pcaF1Line, dpdpcaF1Line, privateF1Line], ['PCA', 'DPDPCA', 'PrivateLocalPCA'], loc=4, fontsize='small');
    """
    #plt.legend([pcaF1Line, dpdpcaF1Line, privateF1Line], ['PCA', 'DPDPCA', 'PrivateLocalPCA'], loc=4, fontsize='small');
    ax.legend((dpdpcaBar[0], privateBar[0], pcaBar[0]), ('DPDPCA', 'PrivateLocalPCA', 'PCA'), loc=1, prop={'size': 7});
    plt.xlabel('Samples at Each Data Owner', fontsize=18);
    #plt.ylabel('Accuracy', fontsize=18);
    plt.title(datasetTitle, fontsize=18);
    plt.xticks(x);

    if type=='Accuracy':
        plt.ylabel('Accuracy', fontsize=18);
        formatter = FuncFormatter(to_percent);
        plt.gca().yaxis.set_major_formatter(formatter);
        plt.gcf().subplots_adjust(left=0.15);
    else:
        plt.ylabel('F1-Score', fontsize=18);
    plt.yticks(np.arange(yMin, 1.05, 0.1));

    """
    ax = plt.gca();
    if x[-1] > 100:
        majorLocator = MultipleLocator(8);
    else:
        majorLocator = MultipleLocator(2);
    #ax.xaxis.set_major_locator(majorLocator);
    """
    if figSavedPath is None:
        plt.show();
    else:
        plt.savefig(figSavedPath + "samples_" + datasetTitle + '_bar.pdf', format='pdf', dpi=1000);


def simulatePrivateLocalPCA(data,targetDimension,epsilon):

    C = data.T.dot(data);
    WishartNoiseMatrix = DiffPrivImpl.SymmWishart(epsilon,data.shape[1]);
    noisyC = C + WishartNoiseMatrix;

    noisyLeftSigVectors,noisySingularValues, noisyRightSigVectors = sparse.linalg.svds(noisyC, k=targetDimension, tol=0.001);
    noisySingularValues = np.real(noisySingularValues);
    noisyRightSigVectors = np.real(noisyRightSigVectors.T);
    S = np.diagflat(np.sqrt(noisySingularValues));
    P = np.dot(noisyRightSigVectors,S);
    return P;

def simulatePrivateGlobalPCA(data,numOfSamples,targetDimension,epsilon):
    numOfCopies = data.shape[0]/numOfSamples;
    dataOwnerGroups = np.array_split(data, numOfCopies);
    P = None;
    for singleDataOwnerCopy in dataOwnerGroups:
        k_prime = min(targetDimension,LA.matrix_rank(singleDataOwnerCopy));
        P_prime = simulatePrivateLocalPCA(singleDataOwnerCopy, k_prime, epsilon);
        if P is not None:
            k_prime = max(k_prime, LA.matrix_rank(P));
            tmpSummary = np.hstack((P_prime, P));
            P = simulatePrivateLocalPCA(tmpSummary.T, k_prime, epsilon);
        else:
            P = P_prime;
    #return P;
    return preprocessing.normalize(P, axis=0, copy=False);

def singleExp(xSamples,trainingData,testingData,topK,epsilon,isLinearSVM):

    pureTrainingData = trainingData[:,1:];
    trainingLabel = trainingData[:,0];

    pureTestingData = testingData[:,1:];
    testingLabel = testingData[:,0];


    scaler = StandardScaler();
    #print pureTrainingData[0];
    #scaler.fit(pureTrainingData);
    pureTrainingData=scaler.fit_transform(pureTrainingData);
    #print pureTrainingData[0];

    #print pureTestingData[0];
    pureTestingData=scaler.transform(pureTestingData);
    #print pureTestingData[0];

    preprocessing.normalize(pureTrainingData, copy=False);
    preprocessing.normalize(pureTestingData, copy=False);

    numOfFeature = trainingData.shape[1]-1;

    pcaImpl = PCAImpl(pureTrainingData);
    pcaImpl.getPCs(topK);
    
    '''
    To get a Wishart Noise projection matrix.
    '''
    WishartNoiseMatrix = DiffPrivImpl.SymmWishart(epsilon,numOfFeature);
    noisyCovMatrix = pcaImpl.covMatrix + WishartNoiseMatrix;

    noisyLeftSigVectors, noisyEigValues,noisyProjMatrix = sparse.linalg.svds(noisyCovMatrix, k=topK, tol=0.001);
    noisyProjMatrix = np.real(noisyProjMatrix.T);

    """
    projTrainingData2 = np.dot(pureTrainingData, noisyProjMatrix);
    projTestingData2 = np.dot(pureTestingData, noisyProjMatrix);

    print "DPDPCA %d" % topK;
    if isLinearSVM:
        result = SVMModule.SVMClf.linearSVM(projTrainingData2, trainingLabel, projTestingData2, testingLabel);
    else:
        result = SVMModule.SVMClf.rbfSVM(projTrainingData2, trainingLabel, projTestingData2, testingLabel);

    cprResult.append(result[2]);
    """

    cprResult = [];

    for k, numOfSamples in np.ndenumerate(xSamples):

        cprResult.append(numOfSamples);
        # Project the data using different projection matrix.
        projTrainingData1 = pcaImpl.transform(pureTrainingData, numOfSamples);
        projTestingData1 = pcaImpl.transform(pureTestingData, numOfSamples);
        print "Non-noise PCA %d" % numOfSamples;
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData1, trainingLabel, projTestingData1, testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData1, trainingLabel, projTestingData1, testingLabel);

        cprResult.extend(result);

        projTrainingData2 = np.dot(pureTrainingData, noisyProjMatrix[:, :numOfSamples]);
        projTestingData2 = np.dot(pureTestingData, noisyProjMatrix[:, :numOfSamples]);

        print "DPDPCA %d" % numOfSamples;
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData2, trainingLabel, projTestingData2, testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData2, trainingLabel, projTestingData2, testingLabel);

        cprResult.extend(result);

        pgProjMatrix = simulatePrivateGlobalPCA(pureTrainingData,numOfSamples,topK,epsilon);

        projTrainingData3 = np.dot(pureTrainingData,pgProjMatrix);
        projTestingData3 = np.dot(pureTestingData,pgProjMatrix);
        
        print "\nPrivateLocalPCA with %d data held by each data owner" % numOfSamples;
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData3,trainingLabel,projTestingData3,testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData3,trainingLabel,projTestingData3,testingLabel);
        cprResult.extend(result);

    resultArray = np.asarray(cprResult);
    resultArray = np.reshape(resultArray, (len(xSamples), -1));
    return resultArray;

def doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfPointsinXAxis, isLinearSVM=True):
    if os.path.basename(datasetPath).endswith('npy'):
        data = np.load(datasetPath);
    else:
        data = np.loadtxt(datasetPath, delimiter=",");
    numOfFeature = data.shape[1] - 1;
    scaler = StandardScaler();
    data_std = scaler.fit_transform(data[:, 1:]);
    globalPCA = PCAImpl(data_std);
    largestReducedFeature = globalPCA.getNumOfPCwithKPercentVariance(varianceRatio);

    print "%d/%d dimensions captures %.2f variance." % (largestReducedFeature,numOfFeature,varianceRatio);
    cprResult = None;


    #rs = StratifiedShuffleSplit(n_splits=numOfRounds, test_size=.2, random_state=0);
    #rs.get_n_splits(data[:,1:],data[:,0]);
    rs = ShuffleSplit(n_splits=numOfRounds, test_size=.2, random_state=0);
    rs.get_n_splits(data);
    for train_index, test_index in rs.split(data):
    #for train_index, test_index in rs.split(data[:,1:],data[:,0]):

        trainingData = data[train_index];
        testingData = data[test_index];
        print "number of training samples %d" % trainingData.shape[0];
        #tmpResult = p.apply_async(singleExp, (xDimensions,trainingData,testingData,topK,isLinearSVM));
        #cprResult += tmpResult.get();
        mostSamplesPerDataOwner = trainingData.shape[0] / 2;
        xSamples = np.arange(2, mostSamplesPerDataOwner, max(mostSamplesPerDataOwner/numOfPointsinXAxis, 1));
        print "number of samples be tested: %s" % xSamples;
        tmpResult = singleExp(xSamples, trainingData, testingData, largestReducedFeature, epsilon, isLinearSVM);
        if cprResult is None:
            cprResult = tmpResult;
        else:
            cprResult = np.concatenate((cprResult,tmpResult),axis=0);

    for result in cprResult:
        print ','.join(['%.3f' % num for num in result]);

    return cprResult;

if __name__ == "__main__":
    
    numOfRounds = 2;
    epsilon = 0.3;
    varianceRatio = 0.8
    #numOfSamples = 2;
    numOfPointsinXAxis = 20;
    figSavedPath = "./fig/";
    resultSavedPath = "./log/";
    isLinearSVM = False;
    if len(sys.argv) > 1:
        datasetPath = sys.argv[1];
        print "+++ using passed in arguments: %s" % (datasetPath);
        result = doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfPointsinXAxis,isLinearSVM=isLinearSVM);
        np.savetxt(resultSavedPath+"dataOwner_"+os.path.basename(datasetPath)+".output",result,delimiter=",",fmt='%1.3f');
    else:
        datasets = ['CNAE','ISOLET','GISETTE'];
        for dataset in datasets:
            print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
            datasetPath = "./input/"+dataset+"_prePCA";
            #result = doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfPointsinXAxis,isLinearSVM=isLinearSVM);
            #np.savetxt(resultSavedPath+"dataOwner_"+dataset+".output",result,delimiter=",",fmt='%1.3f');
            drawScore(dataset, data=None, path=resultSavedPath+"samples_"+dataset+".output",type='F1',figSavedPath=figSavedPath);