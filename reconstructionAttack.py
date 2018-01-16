import numpy as np;
import matplotlib.pyplot as plot;
import scipy;
from pkg.dimReduction import PCAModule;
from pkg.dimReduction import LDAModule;
from sklearn.cluster import KMeans;
from pkg.svm import SVMModule;
from sklearn.model_selection import ShuffleSplit;
from sklearn.preprocessing import StandardScaler;
import matplotlib.pyplot as plt;
from matplotlib.ticker import MultipleLocator;
from scipy.spatial.distance import squareform;
from scipy.spatial.distance import pdist, jaccard, cdist;


def drawResult(datasetTitle, data=None, path=None, figSavedPath=None):
    plt.clf();
    if path is not None:
        data = np.loadtxt(path, delimiter=",");
    x = data[:, 0];
    print "Number of points on x-axis: %d" % len(x);

    # rawDistLine, = plt.plot(x, data[:,1], 'g-');

    weightedDistLine, = plt.plot(x, data[:, 2], 'r-');
    optimizationTargetLine, = plt.plot(x, data[:, 3], 'b-');
    clusterSampleRatioLine = plt.plot(x, data[:, 4], 'p-');
    maxVector = np.amax(data[:, 2:], axis=0);
    yMax = max(maxVector);

    plt.axis([0, x[-1] + 1, 0, yMax + 1]);
    # plt.axis([0,10,0.4,1.0]);
    # plt.legend([rawDistLine, weightedDistLine, optimizationTargetLine], ['raw', 'weighted', 'opTarget'], loc=1);
    plt.xlabel('Number of Clusters', fontsize=18);
    plt.ylabel('dist', fontsize=18);
    plt.title(datasetTitle, fontsize=18);
    plt.xticks(x);
    ax = plt.gca();
    largestXVal = x[-1];
    if largestXVal > 50:
        majorLocator = MultipleLocator(8);
    else:
        majorLocator = MultipleLocator(2);
    ax.xaxis.set_major_locator(majorLocator);
    if figSavedPath is None:
        plt.show();
    else:
        plt.savefig(figSavedPath + "cluster_" + datasetTitle + '.pdf', format='pdf', dpi=1000);


def displayImg(vector):
    tmp = np.reshape(vector, (30, 30));
    plot.gray();
    plot.imshow(tmp);
    plot.show();


def projAndReconstruct(testData, projMatrix, meanVector):
    avgTestData = testData - meanVector;
    # print avgTestData[0];
    projData = np.dot(avgTestData, projMatrix);
    print "Projected data:";
    print projData[0];
    reconstData = np.real(np.dot(projData, projMatrix.T));
    # print "Reconstructed data:";
    # print reconstData[0];
    if reconstData.ndim != 1:
        testDataDist = scipy.spatial.distance.pdist(avgTestData);
        print scipy.spatial.distance.squareform(testDataDist)[0];
        projDataDist = scipy.spatial.distance.pdist(projData);
        print scipy.spatial.distance.squareform(projDataDist)[0];
        reconstDataDist = scipy.spatial.distance.pdist(reconstData);
        print scipy.spatial.distance.squareform(reconstDataDist)[0];

        avgReconstData = np.average(reconstData, axis=0);
        '''
        reconDist = cdist(avgTestData,reconstData,'euclidean');
        diagDist = reconDist.diagonal();
        print np.average(diagDist);

        print "===============================";
        for i in range(0,len(avgTestData)):
            print cdist([avgTestData[i]],[reconstData[i]],'euclidean');
        '''
    else:
        avgReconstData = reconstData;
        print "Euclidean distance: %f" % scipy.spatial.distance.cdist([avgTestData], [reconstData], 'euclidean');
    return (avgReconstData + meanVector).astype(int);


def testWithImg():
    posiPath = "../faceDetection_Android/P/trainingFile";
    posiData = np.genfromtxt(posiPath, delimiter="\t")[:, :-1];

    negPath = "../faceDetection_Android/N/trainingFileNeg";
    negData = np.genfromtxt(negPath, delimiter="\t")[:, :-1];

    composeData = posiData[:20, :];
    composeData = np.append(composeData, negData[range(0, 450, 20), :], axis=0);

    print composeData.shape;
    trainingSim = scipy.spatial.distance.cdist(composeData, composeData, 'cosine');
    print trainingSim[0];

    pcaImpl = PCAModule.PCAImpl(composeData);
    pcaImpl.getPCs();

    energies = pcaImpl.getEigValueEnergies();
    print "Energies preserved:"
    print energies[:20];
    cumEnergies = 0.95;
    tmpSum = 0;
    i = 0;
    while (tmpSum < cumEnergies):
        tmpSum = tmpSum + energies[i];
        i = i + 1;
    print i;
    numOfPC = i;

    projMatrix = pcaImpl.projMatrix[:, 0:numOfPC];
    for i in range(0, 450, 20):
        augmentedTrainingComData = np.append(composeData, [negData[i, :]], axis=0);
        print augmentedTrainingComData.shape;
        augPCAImpl = PCAModule.PCAImpl(augmentedTrainingComData);
        augPCAImpl.getPCs();
        augProjMatrix = augPCAImpl.projMatrix[:, 0:numOfPC];

        print "Mean vector Distance:";
        print scipy.spatial.distance.cdist([pcaImpl.mean], [augPCAImpl.mean], 'cosine');

        print "Augmented eigenvector distance:";
        print scipy.spatial.distance.cdist(projMatrix.T, augProjMatrix.T, 'cosine').diagonal();

    posiTestData = posiData[50:80, :];
    attackPCAImpl = PCAModule.PCAImpl(posiTestData);
    attackPCAImpl.getPCs();
    attackProjMatrix = attackPCAImpl.projMatrix[:, 0:numOfPC];

    simDist = scipy.spatial.distance.cdist(projMatrix.T, attackProjMatrix.T, 'cosine');
    print simDist.diagonal();

    '''
    reconstData = projAndReconstruct(posiTestData,projMatrix,meanVector);
    #displayImg(reconstData);
    print "--------------------------------------";
    #negTestData = negData[451:465,];
    for i in range(451,len(negData),15):
        negTestData = negData[i:i+15,];
        negReconstData = projAndReconstruct(negTestData,projMatrix,meanVector);
        #displayImg(negReconstData);
        print "+++++++++++++++++++++++++++++++++++++++++";
    '''

def findMinPCAinKClusters(trainingData, targetClusters, numOfPCs, projMatrix, energies, totalEnergy):
    kmeans = KMeans(n_clusters=targetClusters, random_state=0, n_jobs=10).fit(trainingData);
    minPCAImpl = None;
    minWeightedDistance = 10000;
    for i in range(targetClusters):
        singleClusterData = trainingData[kmeans.labels_ == i];
        # print "Cluster %d:%d" % (i,singleClusterData.shape[0]);
        sampleIndices = np.where(kmeans.labels_ == i);

        '''
        clusterIndices = np.asarray(np.where(kmeans.labels_ == i));
        #Positive indices intersection and union
        print "Jaccard similarity of positive indices: %f" % calcJaccardSimilarity(clusterIndices[0],posiIndices[0]);

        #Negative indices intersection and union
        print "Jaccard similarity of negative indices: %f" % calcJaccardSimilarity(clusterIndices[0],negIndices[0]);
        '''
        if singleClusterData.shape[0] >= numOfPCs:
            tmpPCAImpl = PCAModule.PCAImpl(singleClusterData);
            tmpPCAImpl.getPCs();
            tmpTotalEnergy = np.sum(tmpPCAImpl.eigValues);
            # print "Total energy of cluster %d data is %f, it takes over %f of the whole data energy." % (i,tmpTotalEnergy,(tmpTotalEnergy/totalEnergy));
            """
            Since the PCs contain possible positive and negative values,
            so the cosine similarity is between -1 and 1,
            then the cosine distance is 1 - cosine similarity, then it ranges
            between 0 and 2.
            """
            simDist = cdist(projMatrix.T[:numOfPCs], tmpPCAImpl.projMatrix.T[:numOfPCs],'cosine');
            # print simDist.diagonal();
            rawSimDist = np.sum(simDist.diagonal());
            # print "Cluster %d Cosine Distance in total: %f" % (i, rawSimDist);
            '''
            approSim = 1 - simDist.diagonal();
            print "Approximate Similarity:"
            print approSim;
            '''
            weightedDistance = 0;
            for j in range(numOfPCs):
                weightedDistance = weightedDistance + simDist.diagonal()[j] * energies[j];

            # print "Weighted Cosine Distance is: %f" % (weightedDistance);
            #optimizationTarget = weightedDistance + (tmpTotalEnergy / totalEnergy);
            sampleRatio = 1.0 * singleClusterData.shape[0] / trainingData.shape[0];
            optimizationTarget = calcOptimizationScore(weightedDistance,sampleRatio);
            # print "Optimization target: %f" % optimizationTarget;

            if minWeightedDistance > weightedDistance:
                minWeightedDistance = weightedDistance;
                minPCAImpl = tmpPCAImpl;
                minClusterIndex = i;
                minRawSimDist = rawSimDist;
                minOptimizationTarget = optimizationTarget;
                minSampleRatio = sampleRatio;
                minSampleIndices = sampleIndices[0];
            # print "\n";
    projTrainingData = minPCAImpl.transform(trainingData, numOfPCs);
    reconstTrainData = minPCAImpl.reconstruct(projTrainingData);
    NMAE = calcNormalizedMeanAbsoluteError(trainingData, reconstTrainData);
    clusterCenterDist = np.sum(np.abs(minPCAImpl.mean));
    print "Minimum cluster index is %d, min raw cosine distance is %f, min weighted cosine distance is %f, min optimization target is %f, NMAE is %f." % (
        minClusterIndex, minRawSimDist, minWeightedDistance, minOptimizationTarget, NMAE);
    """
    for i in range(numOfCluster):
        singleClusterData = pureTrainingData[kmeans.labels_ == i];
        reducedAData = minPCAImpl.transform(singleClusterData, numOfPCs);
        # np.savetxt(aPCAPath+str(i),reducedAData,delimiter=",",fmt='%1.2f');

        reducedGData = pcaImpl.transform(singleClusterData, numOfPCs);
        # np.savetxt(gPCAPath+str(i),reducedGData,delimiter=",",fmt='%1.2f');
        # print pcaImpl.transform(singleClusterData,numOfPCs);

    """
    return [targetClusters, minRawSimDist, minWeightedDistance, minOptimizationTarget, minSampleRatio,
            clusterCenterDist, NMAE], minSampleIndices;


def findMinPCA(data, varianceRatio):
    pureTrainingData = data;
    pcaImpl = PCAModule.PCAImpl(pureTrainingData);
    pcaImpl.getPCs();

    energies = pcaImpl.getEigValueEnergies();
    numOfPCs = pcaImpl.getNumOfPCwithKPercentVariance(varianceRatio);
    projMatrix = pcaImpl.projMatrix;
    totalEnergy = np.sum(pcaImpl.eigValues);
    print "The total eigenvalue energies is %f, to achieve %f percentage, it needs %d principal components." % (totalEnergy, varianceRatio, numOfPCs);

    numOfCluster = pureTrainingData.shape[0] / numOfPCs;
    print "Maximum number of clusters; %d" % numOfCluster;
    print "global PCA performance on classification"
    '''
    for i in range(len(composeData)):
        print "%d , %d" % (i,kmeans.labels_[i]);
    gPCAPath = "./"+subject+"_c";
    aPCAPath = "./"+subject+"_appro_c";
    '''
    clusterRes = [];
    clusterSampleIndices = [];
    numOfCluster = 20;
    for j in range(2, numOfCluster + 1):
        print "%d-means" % j;
        res, minSampleIndices = findMinPCAinKClusters(pureTrainingData, j, numOfPCs, projMatrix, energies, totalEnergy);

        clusterRes.append(res);
        clusterSampleIndices.append(minSampleIndices);

    clusterResArray = np.asarray(clusterRes);
    # Using optimizationTarget to find the best candidate subset.
    sortedWeightedDistIndices = np.argsort(clusterResArray[:, 3]);
    # print sortedWeightedDistIndices;
    sortedResArray = clusterResArray[sortedWeightedDistIndices];
    for res in sortedResArray:
        print "%d,%f,%f,%f,%f,%f,%f" % (res[0], res[1], res[2], res[3], res[4], res[5], res[6]);

    sampleIndicesArray = np.asarray(clusterSampleIndices);
    sortedClusterSampleIndices = sampleIndicesArray[sortedWeightedDistIndices];
    """
    # Print the Jaccard similarity matrix.
    jacSimMat = np.zeros((sortedClusterSampleIndices.shape[0], sortedClusterSampleIndices.shape[0]));
    for i in range(sortedClusterSampleIndices.shape[0]):
        for j in range(i, sortedClusterSampleIndices.shape[0]):
            jacSimMat[i, j] = calcJaccardSimilarity(sortedClusterSampleIndices[i], sortedClusterSampleIndices[j]);
    print jacSimMat[:5, :5];
    """
    return sortedClusterSampleIndices[0];


def binaryLDAExp(path, numOfRounds, varianceRatio, subject):
    print "*************** %s ****************" % path;
    data = np.loadtxt(path, delimiter=",");
    rs = ShuffleSplit(n_splits=numOfRounds, test_size=.1, random_state=0);
    rs.get_n_splits(data);
    for train_index, test_index in rs.split(data):

        trainingData = data[train_index];

        scaler = StandardScaler(copy=True);
        scaler.fit(trainingData[:, 1:]);
        pureTrainingData = scaler.transform(trainingData[:,1:]);
        pureTestingData = scaler.transform(data[test_index, 1:]);

        posiIndices = np.where(trainingData[:, 0] == 1)[0];
        negIndices = np.where(trainingData[:, 0] == -1)[0];
        # print posiIndices;

        print "Training Samples: %d, dimension: %d" % (pureTrainingData.shape[0], pureTrainingData.shape[1]);

        ldaImpl = LDAModule.LDAImpl(pureTrainingData,trainingData[:,0]);
        ldaImpl.getPCs();
        numOfClass = ldaImpl.numOfClass;
        numOfCluster = pureTrainingData.shape[0] / pureTrainingData.shape[1];
        print "Maximum number of clusters; %d" % numOfCluster;
        """
        print "global LDA performance on classification"
        oriPCAReducedData = ldaImpl.transform(pureTrainingData, numOfClass );
        testOriPCAReducedData = ldaImpl.transform(pureTestingData, numOfClass);
        SVMModule.SVMClf.rbfSVM(oriPCAReducedData, data[train_index, 0], testOriPCAReducedData, data[test_index, 0]);
        """
        print "Positive min PCA"
        posTrainingData = pureTrainingData[posiIndices];
        posMinPCAIndices = findMinPCA(posTrainingData,varianceRatio);
        print "Negative min PCA"
        negTrainingData = pureTrainingData[negIndices];
        negMinPCAIndices = findMinPCA(negTrainingData,varianceRatio);

        posiLabels = trainingData[posiIndices,0];
        negLabels = trainingData[negIndices,0];

        minTrainingData = np.concatenate((posTrainingData[posMinPCAIndices],negTrainingData[negMinPCAIndices]),axis=0);
        minTrainingLabel = np.concatenate((posiLabels[posMinPCAIndices],negLabels[negMinPCAIndices]));
        minLDAImpl = LDAModule.LDAImpl(minTrainingData,minTrainingLabel);
        minLDAImpl.getPCs();
        simDist = cdist(ldaImpl.projMatrix.T,minLDAImpl.projMatrix.T, 'cosine');
        print "The cosine distance in LDA experiment : %s" % simDist;

        print "SVM with global LDA"
        oriLDAReducedData = ldaImpl.transform(pureTrainingData, 2);
        testOriLDAReducedData = ldaImpl.transform(pureTestingData, 2);
        SVMModule.SVMClf.rbfSVM(oriLDAReducedData, data[train_index, 0], testOriLDAReducedData, data[test_index, 0]);

        print "SVM with min LDA"
        approLDAReducedData = minLDAImpl.transform(pureTrainingData, 2);
        testApproLDAReducedData = minLDAImpl.transform(pureTestingData, 2);
        SVMModule.SVMClf.rbfSVM(approLDAReducedData, data[train_index, 0], testApproLDAReducedData,data[test_index, 0]);


def PCAExp(path, numOfRounds, varianceRatio, subject):
    print "*************** %s ****************" % path;
    data = np.loadtxt(path, delimiter=",");
    rs = ShuffleSplit(n_splits=numOfRounds, test_size=.1, random_state=0);
    rs.get_n_splits(data);
    for train_index, test_index in rs.split(data):

        posiIndices = np.asarray(np.where(data[:, 0] == 1));
        negIndices = np.asarray(np.where(data[:, 0] == -1));
        # print posiIndices;

        scaler = StandardScaler(copy=True);
        scaler.fit(data[train_index, 1:]);
        pureTrainingData = scaler.transform(data[train_index, 1:]);
        pureTestingData = scaler.transform(data[test_index, 1:]);
        print "Training Samples: %d, dimension: %d" % (pureTrainingData.shape[0], pureTrainingData.shape[1]);
        # composeData = data[:,1:];

        minPCAIndices = findMinPCA(pureTrainingData,varianceRatio);

        pcaImpl = PCAModule.PCAImpl(pureTrainingData);
        pcaImpl.getPCs();
        numOfPCs = pcaImpl.getNumOfPCwithKPercentVariance(varianceRatio);
        print "SVM with Global PCA"
        oriPCAReducedData = pcaImpl.transform(pureTrainingData, numOfPCs);
        testOriPCAReducedData = pcaImpl.transform(pureTestingData, numOfPCs);
        SVMModule.SVMClf.rbfSVM(oriPCAReducedData, data[train_index, 0], testOriPCAReducedData, data[test_index, 0]);

        print "SVM with min PCA"
        minPCAImpl = PCAModule.PCAImpl(pureTrainingData[minPCAIndices]);
        minPCAImpl.getPCs();
        approPCAReducedData = minPCAImpl.transform(pureTrainingData, numOfPCs);
        testApproPCAReducedData = minPCAImpl.transform(pureTestingData, numOfPCs);
        SVMModule.SVMClf.rbfSVM(approPCAReducedData, data[train_index, 0], testApproPCAReducedData,data[test_index, 0]);

        """
        pcaImpl = PCAModule.PCAImpl(pureTrainingData);
        pcaImpl.getPCs();

        energies = pcaImpl.getEigValueEnergies();
        numOfPCs = pcaImpl.getNumOfPCwithKPercentVariance(varianceRatio);
        projMatrix = pcaImpl.projMatrix;
        totalEnergy = np.sum(pcaImpl.eigValues);
        print "The total eigenvalue energies is %f, to achieve %f percentage, it needs %d principal components." % (
        totalEnergy, varianceRatio, numOfPCs);

        numOfCluster = pureTrainingData.shape[0] / numOfPCs;
        print "Maximum number of clusters; %d" % numOfCluster;
        print "global PCA performance on classification"
        oriPCAReducedData = pcaImpl.transform(pureTrainingData, numOfPCs);
        testOriPCAReducedData = pcaImpl.transform(pureTestingData, numOfPCs);
        SVMModule.SVMClf.rbfSVM(oriPCAReducedData, data[train_index, 0], testOriPCAReducedData, data[test_index, 0]);
        '''
        for i in range(len(composeData)):
            print "%d , %d" % (i,kmeans.labels_[i]);
        gPCAPath = "./"+subject+"_c";
        aPCAPath = "./"+subject+"_appro_c";
        '''
        clusterRes = [];
        clusterSampleIndices = [];
        numOfCluster = 20;
        for j in range(2, numOfCluster + 1):
            print "%d-means" % j;
            res, minSampleIndices = findMinPCAinKClusters(pureTrainingData, j, numOfPCs, projMatrix, energies, totalEnergy);
            clusterRes.append(res);
            clusterSampleIndices.append(minSampleIndices);

            tmpMinPCAImpl = PCAModule.PCAImpl(pureTrainingData[minSampleIndices]);
            tmpMinPCAImpl.getPCs();
            approPCAReducedData = tmpMinPCAImpl.transform(pureTrainingData, numOfPCs);
            # print composeData.shape;
            testApproPCAReducedData = tmpMinPCAImpl.transform(pureTestingData, numOfPCs);
            # print testData[:,1:].shape;
            # print("=====================================");
            SVMModule.SVMClf.rbfSVM(approPCAReducedData, data[train_index, 0], testApproPCAReducedData,
                                    data[test_index, 0]);

        clusterResArray = np.asarray(clusterRes);
        sortedWeightedDistIndices = np.argsort(clusterResArray[:, 2]);
        # print sortedWeightedDistIndices;
        sortedResArray = clusterResArray[sortedWeightedDistIndices];
        for res in sortedResArray:
            print "%d,%f,%f,%f,%f,%f,%f" % (res[0], res[1], res[2], res[3], res[4], res[5], res[6]);

        sampleIndicesArray = np.asarray(clusterSampleIndices);
        sortedClusterSampleIndices = sampleIndicesArray[sortedWeightedDistIndices];
        jacSimMat = np.zeros((sortedClusterSampleIndices.shape[0], sortedClusterSampleIndices.shape[0]));
        for i in range(sortedClusterSampleIndices.shape[0]):
            for j in range(i, sortedClusterSampleIndices.shape[0]):
                jacSimMat[i, j] = calcJaccardSimilarity(sortedClusterSampleIndices[i], sortedClusterSampleIndices[j]);
        print jacSimMat[:5, :5];
        # print wholeArray;
        """

def calcJaccardSimilarity(xa, xb):
    intersectionIndices = np.intersect1d(xa, xb, True);
    unionIndices = np.union1d(xa, xb);
    return (1.0 * len(intersectionIndices) / len(unionIndices));


def calcNormalizedMeanAbsoluteError(originalData, reconstructedData):
    diffMat = reconstructedData - originalData;
    resMat = np.abs(np.divide(diffMat, originalData));
    colSum = np.sum(resMat, axis=0);
    res = np.sum(colSum) / (originalData.shape[0] * originalData.shape[1]);
    return res;
def calcOptimizationScore(weightedDistance,sampleRatio,alpha=1):
    alphaSquare = alpha*alpha;
    optScore = (1+alphaSquare)*weightedDistance*sampleRatio/(alphaSquare*weightedDistance+sampleRatio);
    return optScore;
def testKMeans_GroundTruth():
    posiPath = "../faceDetection_Android/P/trainingFile";
    posiData = np.genfromtxt(posiPath, delimiter="\t")[:, :-1];

    negPath = "../faceDetection_Android/N/trainingFileNeg";
    negData = np.genfromtxt(negPath, delimiter="\t")[:, :-1];

    composeData = posiData[:20, :];
    composeData = np.append(composeData, negData[range(0, 450, 20), :], axis=0);

    meanVector = np.mean(composeData, axis=0);

    avgComposeData = composeData - meanVector;
    pcaImpl = PCAModule.PCAImpl(avgComposeData);
    pcaImpl.getPCs();
    projMatrix = pcaImpl.projMatrix;

    energies = pcaImpl.getEigValueEnergies();
    cumEnergies = 0.95;
    tmpSum = 0;
    i = 0;
    while (tmpSum < cumEnergies):
        tmpSum = tmpSum + energies[i];
        i = i + 1;
    print i;
    print "To achieve %f energies, it needs %d principal components." % (cumEnergies, i);
    numOfPCs = i;
    avgPoint = len(composeData);
    minPoint = avgPoint / 2.0;
    minIndex = 0;
    pointList = [];
    while (minPoint < avgPoint):
        for i in range(len(composeData)):
            tmpIndices = range(len(composeData));
            tmpIndices = np.delete(tmpIndices, i);
            # print tmpIndices;
            leaveOneOutData = composeData[tmpIndices];
            meanVector = np.mean(leaveOneOutData, axis=0);
            avgComposeData = leaveOneOutData - meanVector;
            pcaImpl = PCAModule.PCAImpl(avgComposeData);
            pcaImpl.getPCs();
            simDist = scipy.spatial.distance.cdist(projMatrix.T[:numOfPCs], pcaImpl.projMatrix.T[:numOfPCs], 'cosine');
            simDistTotal = np.sum(simDist.diagonal());
            print "Cluster %d Cosine Distance in total: %f" % (i, simDistTotal);
            # print simDist.diagonal();
            # print "\n";
            if minPoint > simDistTotal:
                minPoint = simDistTotal;
                minIndex = i;
                print "Min Index is %d, and min point is %f" % (minIndex, minPoint);
            pointList.append(simDistTotal);
        avgPoint = np.average(pointList);
        minPoint = avgPoint;
        composeData = np.delete(composeData, minIndex, 0);

    '''
def genSyntheticData(dim,covDiag,size):
    mean = np.zeros(dim);
    cov = np.diagflat(covDiag);
    synData = np.random.multivariate_normal(mean,cov,size);
    return synData;
def testWithSyntheticData():
    dim = 10;
    size = 50;
    upperBound = 100;
    covDiag = np.random.randint(upperBound, size=dim);
    synData = genSyntheticData(dim,covDiag,size);
    #print synData[:,0];
    pcaImpl = PCAModule.PCAImpl(synData);
    pcaImpl.getPCs();
    projMatrix = pcaImpl.projMatrix;

    synData2 = genSyntheticData(dim,covDiag,size);
    pcaImpl2 = PCAModule.PCAImpl(synData2);
    pcaImpl2.getPCs();
    projMatrix2 = pcaImpl2.projMatrix;

    simDist = scipy.spatial.distance.cdist(projMatrix.T,projMatrix2.T,'cosine');
    print simDist.diagonal();
    '''


if __name__ == "__main__":
    # testWithSyntheticData();
    # testKMeans_GroundTruth();
    '''
    posiPath = "../faceDetection_Android/P/trainingFile";
    posiData = np.genfromtxt(posiPath,delimiter="\t")[:,:-1];
    negPath = "../faceDetection_Android/N/trainingFileNeg";
    negData = np.genfromtxt(negPath,delimiter="\t")[:,:-1];

    colPOnes = np.ones((len(posiData),1));
    colNOnes = np.ones((len(negData),1));
    colNOnes = -1*colNOnes;
    posiData = np.append(colPOnes,posiData,axis=1);
    negData = np.append(colNOnes,negData,axis=1);

    trainingData = posiData[:150,:];
    trainingData = np.append(trainingData,negData[:200,:],axis=0);
    testingData = posiData[150:,:];
    testingData = np.append(testingData,negData[200:,:],axis=0);
    print trainingData.shape;

    np.savetxt("./face_prePCA_training",trainingData,delimiter=",",fmt='%d');
    np.savetxt("./face_prePCA_testing",testingData,delimiter=",",fmt='%d');
    '''
    varianceRatio = 0.9;
    numOfRounds = 1;

    subject = "diabetes";
    path = "./input/" + subject + "_prePCA";
    resPath = "./log/privateSubspace/" + subject + ".output";
    #PCAExp(path, numOfRounds, varianceRatio, subject);
    binaryLDAExp(path, numOfRounds, varianceRatio, subject);
    # drawResult(subject,None,resPath,None);