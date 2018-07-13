from sklearn.model_selection import StratifiedShuffleSplit;
from sklearn.preprocessing import StandardScaler;
from sklearn import preprocessing;
from sklearn.random_projection import GaussianRandomProjection;

import numpy as np;
from numpy import linalg as LA;
import sys;
import os;

from pkg import SVMModule;
from pkg.DimReduction import PCAImpl;
from pkg.DPDimReduction import DiffPrivPCAImpl;
from pkg.DPDimReduction import DiffPrivImpl,DPPro;
from pkg.global_functions import globalFunction as gf;


def drawFig(datasetTitle, data=None, path=None, n_trails=1, type='f1',figSavedPath=None):
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
    elif type is 'precision':
        #drawPrecision(datasetTitle, data=data, path=path, figSavedPath=figSavedPath)
        data_mean = data_mean[:, [1, 5, 9]];
        data_std = data_std[:, [1, 5, 9]];
    elif type is 'recall':
        #drawRecall(datasetTitle, data=data, path=path, figSavedPath=figSavedPath)
        data_mean = data_mean[:, [2, 6, 10]];
        data_std = data_std[:, [2, 6, 10]];

    minVector = np.amin(data_mean, axis=0);
    yMin = min(minVector);
    maxVector = np.amax(data_mean, axis=0);
    yMax = max(maxVector);

    yMin = (yMin - 0.05) if (yMin - 0.05) > 0 else 0;
    yMax = (yMax * 1.35) if (yMax * 1.35) < 1 else 1.05;

    if type is 'accuracy':
        pcaAccErrorLine = plt.errorbar(x, data_mean[:,0], yerr=data_std[:,0], fmt='b', capsize=4);
        pcaAccLine, = plt.plot(x, data_mean[:,0], 'b-')

        dpdpcaErrorLine = plt.errorbar(x,data_mean[:,1], yerr=data_std[:,1], fmt='m', capsize=4);
        dpdpcaLine, = plt.plot(x, data_mean[:,1], 'm-')

        dpproErrorLine = plt.errorbar(x, data_mean[:,2], yerr=data_std[:,2], fmt='g', capsize=4);
        dpproLine, = plt.plot(x, data_mean[:,2], 'g-')

        #plt.axis([0.05, x[-1] + 0.05, yMin, yMax]);
        plt.axis([0.05, x[-1] + 0.05, 0, 1]);
        # plt.axis([0,10,0.4,1.0]);
        plt.legend([pcaAccLine, dpdpcaLine, dpproLine], ['PCA', 'DPDPCA', 'DPPRO'], loc=1);
        # plt.legend([dpdpcaLine, dpproLine], ['DPDPCA', 'DPPRO'], loc=1);

        plt.xlabel('Epsilon', fontsize=18);
        plt.ylabel('Classification Accuracy', fontsize=18);
        plt.title(datasetTitle, fontsize=18);
        plt.xticks(x);

        formatter = FuncFormatter(to_percent);
        plt.gca().yaxis.set_major_formatter(formatter);
        plt.gcf().subplots_adjust(left=0.15);
    else:
        theRange = [1,3,5,7,9];
        x = x[theRange];
        data_mean = data_mean[theRange,:];
        data_std = data_std[theRange,:];

        ax = plt.gca();
        width = 0.05;
        gBar = ax.bar(x - 0.03, data_mean[:,1], width, color='m', yerr=data_std[:,1], capsize=2);
        # gBar = ax.bar(x-0.035, gF1Mean, width, color='r', yerr=gF1Std,capsize=2);
        # wBar = ax.bar(x, wF1Mean, width, color='g', yerr=wF1Std,capsize=2);
        # pcaBar = ax.bar(x+0.035, pcaF1Mean/2, width, color='b', yerr=pcaF1Std,capsize=2);
        pcaBar = ax.bar(x + 0.03, data_mean[:,0], width, color='b', yerr=data_std[:,0], capsize=2);
        plt.axis([0.12, 1.08, 0, 1.15]);
        # plt.legend([pcaF1Line,gF1Line,wF1Line], ['PCA', 'Gaussian Noise', 'Wishart Noise'], loc=4);
        # ax.legend((gBar[0], wBar[0], pcaBar[0]), ('Gaussian Noise','Wishart Noise','PCA'), loc=1, prop={'size':6});
        ax.legend((gBar[0], pcaBar[0]), ('DPDPCA', 'PCA'), loc=1, prop={'size': 8});
        # plt.axis([0,10,0.4,1.0]);
        plt.xlabel('Epsilon', fontsize=18);
        plt.ylabel(type, fontsize=18);
        plt.title(datasetTitle, fontsize=18);
        plt.xticks(x);

    if figSavedPath is None:
        plt.show();
    else:
        plt.savefig(figSavedPath + "dppro_" + datasetTitle + '.pdf', format='pdf', dpi=1000);

def singleExp(xEpsilons,trainingData,testingData,largestReducedFeature,isLinearSVM):
    pureTrainingData = trainingData[:, 1:];
    trainingLabel = trainingData[:, 0];

    numOfTrainingSamples = trainingData.shape[0];

    pureTestingData = testingData[:, 1:];
    testingLabel = testingData[:, 0];

    scaler = StandardScaler();
    # print pureTrainingData[0];
    #scaler.fit(pureTrainingData);
    pureTrainingData=scaler.fit_transform(pureTrainingData);
    # print pureTrainingData[0];

    # print pureTestingData[0];
    pureTestingData=scaler.transform(pureTestingData);
    # print pureTestingData[0];
    
    pcaImpl = PCAImpl(pureTrainingData);
    pcaImpl.getPCs(largestReducedFeature);
    
    dpGaussianPCAImpl = DiffPrivPCAImpl(pureTrainingData);
    dpWishartPCAImpl = DiffPrivPCAImpl(pureTrainingData);
    
    delta = np.divide(1.0,numOfTrainingSamples);
    projTrainingData = pcaImpl.transform(pureTrainingData,largestReducedFeature);
    projTestingData = pcaImpl.transform(pureTestingData,largestReducedFeature);
    #print projTrainingData.shape;
    cprResult = [];
    print "non noise PCA SVM training";
    if isLinearSVM:
        pcaResult = SVMModule.SVMClf.linearSVM(projTrainingData,trainingLabel,projTestingData,testingLabel);
    else:
        pcaResult = SVMModule.SVMClf.rbfSVM(projTrainingData,trainingLabel,projTestingData,testingLabel);

    randomProjector = GaussianRandomProjection(n_components=largestReducedFeature);
    randomProjector.fit(pureTrainingData);

    for k, targetEpsilon in np.ndenumerate(xEpsilons):
        #print pcaImpl.projMatrix[:,0];    
        print "epsilon: %.2f, delta: %f" % (targetEpsilon,delta);
        cprResult.append(targetEpsilon);
        isGaussianDist = True;
        dpGaussianPCAImpl.setEpsilonAndGamma(targetEpsilon,delta);
        dpGaussianPCAImpl.getDiffPrivPCs(isGaussianDist,largestReducedFeature);
        
        isGaussianDist = False;
        dpWishartPCAImpl.setEpsilonAndGamma(targetEpsilon,delta);
        dpWishartPCAImpl.getDiffPrivPCs(isGaussianDist,largestReducedFeature);
        '''
        We don't need to project the data multiple times.
        '''
        cprResult.extend(pcaResult);

        projTrainingData = dpGaussianPCAImpl.transform(pureTrainingData,largestReducedFeature);
        projTestingData = dpGaussianPCAImpl.transform(pureTestingData,largestReducedFeature);
        print "Gaussian-DPDPCA SVM training";
        
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData,trainingLabel,projTestingData,testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData,trainingLabel,projTestingData,testingLabel);
        cprResult.extend(result);

        projTrainingData = dpWishartPCAImpl.transform(pureTrainingData,largestReducedFeature);
        projTestingData = dpWishartPCAImpl.transform(pureTestingData,largestReducedFeature);
        #print projTestingData.shape;
        print "Wishart-DPPCA SVM training";
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData,trainingLabel,projTestingData,testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData,trainingLabel,projTestingData,testingLabel);
        cprResult.extend(result);

        projTrainingData, projTestingData = DPPro(pureTrainingData, pureTestingData, largestReducedFeature, targetEpsilon,randomProjector=randomProjector);
        # print projTestingData.shape;
        print "DPPro SVM training";
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData, trainingLabel, projTestingData, testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData, trainingLabel, projTestingData, testingLabel);
        cprResult.extend(result);
    cprResult = np.asarray(cprResult);
    return cprResult.reshape((len(xEpsilons), -1));


def doExp(datasetPath,varianceRatio,xEpsilons,n_trails,logPath,isLinearSVM=True):
    if os.path.basename(datasetPath).endswith('npy'):
        data = np.load(datasetPath);
    else:
        data = np.loadtxt(datasetPath, delimiter=",");

    scaler = StandardScaler();
    data_std_scale = scaler.fit_transform(data[:, 1:]);
    globalPCA = PCAImpl(data_std_scale);
    n_feature = data.shape[1]-1;
    largestReducedFeature = globalPCA.getNumOfPCwithKPercentVariance(varianceRatio);
    print "%d/%d dimensions captures %.2f variance." % (largestReducedFeature,n_feature,varianceRatio);

    cprResult = [];
    rs = StratifiedShuffleSplit(n_splits=n_trails, test_size=.2, random_state=0);
    rs.get_n_splits(data[:,1:],data[:,0]);
    for train_index, test_index in rs.split(data[:,1:],data[:,0]):
        trainingData = data[train_index];
        testingData = data[test_index];
        tmpResult = singleExp(xEpsilons,trainingData,testingData,largestReducedFeature,isLinearSVM);

        with open(logPath, "a") as f:
            np.savetxt(f,tmpResult,delimiter=",",fmt='%1.3f');
        cprResult.append(tmpResult);

    cprResult = np.vstack(cprResult);
    for result in cprResult:
        print ','.join(['%.3f' % num for num in result]);
    return cprResult;


if __name__ == "__main__":
    
    n_trails = 1;
    varianceRatio = 0.95;
    xEpsilons = np.arange(0.1,1.1,0.1);
    figSavedPath = './fig/';
    resultSavedPath = './log/';
    logSavedPath = './log/';
    isLinearSVM = True;

    if len(sys.argv) > 1:
        datasetPath = sys.argv[1];
        print "+++ using passed in arguments: %s" % (datasetPath);
        result = doExp(datasetPath,varianceRatio,xEpsilons,n_trails,logPath=resultSavedPath+"Epsilon_"+os.path.basename(datasetPath)+".output",isLinearSVM=isLinearSVM);
        #np.savetxt(resultSavedPath+"Epsilon_"+os.path.basename(datasetPath)+".output",result,delimiter=",",fmt='%1.3f');
    else:
        #datasets = ['diabetes','CNAE_2','ionosphere','CNAE_5','CNAE_7','face2','Amazon_3','madelon'];
        datasets = ['Australian','CNAE_2','spokenLetter_A','YaleB','NLTCS-Money','NLTCS-Travel','NLTCS-Outside'];
        for dataset in datasets:
            print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
            datasetPath = "./input/"+dataset+"_prePCA";
            result = doExp(datasetPath,varianceRatio,xEpsilons,n_trails,logPath=logSavedPath+'eps_'+dataset+".out",isLinearSVM=isLinearSVM);
            #np.savetxt(resultSavedPath+"Epsilon_"+dataset+".output",result,delimiter=",",fmt='%1.3f');
            #drawFig(dataset, data=None, path=resultSavedPath + "Epsilon_" + dataset + ".csv",n_trails=10,type='precision',figSavedPath=None);
