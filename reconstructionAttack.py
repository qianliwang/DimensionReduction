import numpy as np;
import matplotlib.pyplot as plot;
import scipy.spatial.distance;
from pkg.dimReduction import PCAModule;
from sklearn.cluster import KMeans;

def displayImg(vector):
    tmp = np.reshape(vector,(30,30));
    plot.gray();
    plot.imshow(tmp);
    plot.show();

def projAndReconstruct(testData,projMatrix,meanVector):
    avgTestData = testData - meanVector;
    #print avgTestData[0];
    projData = np.dot(avgTestData,projMatrix);
    print "Projected data:";
    print projData[0];
    reconstData = np.real(np.dot(projData,projMatrix.T));
    #print "Reconstructed data:";
    #print reconstData[0];
    if reconstData.ndim!=1:
        testDataDist = scipy.spatial.distance.pdist(avgTestData);
        print scipy.spatial.distance.squareform(testDataDist)[0];
        projDataDist = scipy.spatial.distance.pdist(projData);
        print scipy.spatial.distance.squareform(projDataDist)[0];
        reconstDataDist = scipy.spatial.distance.pdist(reconstData);
        print scipy.spatial.distance.squareform(reconstDataDist)[0];
       
        avgReconstData = np.average(reconstData,axis=0);
        '''
        reconDist = scipy.spatial.distance.cdist(avgTestData,reconstData,'euclidean');
        diagDist = reconDist.diagonal();
        print np.average(diagDist);
        
        print "===============================";
        for i in range(0,len(avgTestData)):
            print scipy.spatial.distance.cdist([avgTestData[i]],[reconstData[i]],'euclidean');
        '''
    else:
        avgReconstData = reconstData;
        print "Euclidean distance: %f" % scipy.spatial.distance.cdist([avgTestData],[reconstData],'euclidean');
    return (avgReconstData+meanVector).astype(int);

def testWithImg():
    posiPath = "../faceDetection_Android/P/trainingFile";
    posiData = np.genfromtxt(posiPath,delimiter="\t")[:,:-1];
    
    negPath = "../faceDetection_Android/N/trainingFileNeg";
    negData = np.genfromtxt(negPath,delimiter="\t")[:,:-1];
    
    composeData = posiData[:20,:];
    composeData = np.append(composeData,negData[range(0,450,20),:],axis=0);
    
    print composeData.shape;
    trainingSim = scipy.spatial.distance.cdist(composeData,composeData,'cosine');
    print trainingSim[0];
    meanVector = np.mean(composeData,axis=0);
    
    avgComposeData = composeData - meanVector;
    pcaImpl = PCAModule.PCAImpl(avgComposeData);
    pcaImpl.getPCs();
    
    energies = pcaImpl.getEigValueEnergies();
    print "Energies preserved:"
    print energies[:20];
    cumEnergies = 0.95;
    tmpSum = 0;
    i=0;
    while (tmpSum<cumEnergies):
        tmpSum = tmpSum + energies[i];
        i = i+1;
    print i;
    numOfPC = i;
    
    projMatrix = pcaImpl.projMatrix[:,0:numOfPC];
    for i in range(0,450,20):
        augmentedTrainingComData = np.append(composeData,[negData[i,:]],axis=0);
        print augmentedTrainingComData.shape;
        augMeanVector = np.mean(augmentedTrainingComData,axis=0);
        augPCAImpl = PCAModule.PCAImpl(augmentedTrainingComData-augMeanVector);
        augPCAImpl.getPCs();
        augProjMatrix = augPCAImpl.projMatrix[:,0:numOfPC];
        
        print "Mean vector Distance:";
        print scipy.spatial.distance.cdist([meanVector],[augMeanVector],'cosine');
        
        print "Augmented eigenvector distance:";
        print scipy.spatial.distance.cdist(projMatrix.T,augProjMatrix.T,'cosine').diagonal();
        
    posiTestData = posiData[50:80,:];
    attackPCAImpl = PCAModule.PCAImpl(posiTestData-meanVector);
    attackPCAImpl.getPCs();
    attackProjMatrix = attackPCAImpl.projMatrix[:,0:numOfPC];
    
    simDist = scipy.spatial.distance.cdist(projMatrix.T,attackProjMatrix.T,'cosine');
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
def testKMeans():
    posiPath = "../faceDetection_Android/P/trainingFile";
    posiData = np.genfromtxt(posiPath,delimiter="\t")[:,:-1];
    
    negPath = "../faceDetection_Android/N/trainingFileNeg";
    negData = np.genfromtxt(negPath,delimiter="\t")[:,:-1];
    
    composeData = posiData[:20,:];
    composeData = np.append(composeData,negData[range(0,450,20),:],axis=0);
    
    meanVector = np.mean(composeData,axis=0);
    
    avgComposeData = composeData - meanVector;
    pcaImpl = PCAModule.PCAImpl(avgComposeData);
    pcaImpl.getPCs();
    projMatrix = pcaImpl.projMatrix;
    
    energies = pcaImpl.getEigValueEnergies();
    cumEnergies = 0.8;
    tmpSum = 0;
    i=0;
    while (tmpSum<cumEnergies):
        tmpSum = tmpSum + energies[i];
        i = i+1;
    print i;
    print "To achieve %f energies, it needs %d principal components." % (cumEnergies,i);
    
    numOfPCs = i;
    numOfCluster = 4;
    kmeans = KMeans(n_clusters=numOfCluster, random_state=0, n_jobs=5).fit(composeData);
    '''
    for i in range(len(composeData)):
        print "%d , %d" % (i,kmeans.labels_[i]);
    '''
    for i in range(numOfCluster):
        singleClusterData = composeData[kmeans.labels_ == i];
        print "Cluster %d:%d" % (i,singleClusterData.shape[0]);
        print np.where(kmeans.labels_ == i);
        meanVector = np.mean(singleClusterData,axis=0);
        avgComposeData = singleClusterData - meanVector;
        pcaImpl = PCAModule.PCAImpl(avgComposeData);
        pcaImpl.getPCs();
        simDist = scipy.spatial.distance.cdist(projMatrix.T[:numOfPCs],pcaImpl.projMatrix.T[:numOfPCs],'cosine');
        print "Cluster %d Cosine Distance in total: %f" % (i, np.sum(simDist.diagonal()));
        print simDist.diagonal();
        print "\n";

def testKMeans_GroundTruth():
    posiPath = "../faceDetection_Android/P/trainingFile";
    posiData = np.genfromtxt(posiPath,delimiter="\t")[:,:-1];
    
    negPath = "../faceDetection_Android/N/trainingFileNeg";
    negData = np.genfromtxt(negPath,delimiter="\t")[:,:-1];
    
    composeData = posiData[:20,:];
    composeData = np.append(composeData,negData[range(0,450,20),:],axis=0);
    
    meanVector = np.mean(composeData,axis=0);
    
    avgComposeData = composeData - meanVector;
    pcaImpl = PCAModule.PCAImpl(avgComposeData);
    pcaImpl.getPCs();
    projMatrix = pcaImpl.projMatrix;
    
    energies = pcaImpl.getEigValueEnergies();
    cumEnergies = 0.8;
    tmpSum = 0;
    i=0;
    while (tmpSum<cumEnergies):
        tmpSum = tmpSum + energies[i];
        i = i+1;
    print i;
    print "To achieve %f energies, it needs %d principal components." % (cumEnergies,i);
    numOfPCs = i;
    avgPoint = len(composeData);
    minPoint = avgPoint/2.0;
    minIndex = 0;
    pointList = [];
    while(minPoint<avgPoint):
        for i in range(len(composeData)):
            tmpIndices = range(len(composeData));
            tmpIndices = np.delete(tmpIndices,i);
            #print tmpIndices;
            leaveOneOutData = composeData[tmpIndices];
            meanVector = np.mean(leaveOneOutData,axis=0);
            avgComposeData = leaveOneOutData - meanVector;
            pcaImpl = PCAModule.PCAImpl(avgComposeData);
            pcaImpl.getPCs();
            simDist = scipy.spatial.distance.cdist(projMatrix.T[:numOfPCs],pcaImpl.projMatrix.T[:numOfPCs],'cosine');
            simDistTotal = np.sum(simDist.diagonal());
            print "Cluster %d Cosine Distance in total: %f" % (i, simDistTotal);
            #print simDist.diagonal();
            #print "\n";
            if minPoint>simDistTotal:
                minPoint = simDistTotal;
                minIndex = i;
                print "Min Index is %d, and min point is %f" % (minIndex,minPoint);
            pointList.append(simDistTotal);
        avgPoint = np.average(pointList);
        minPoint = avgPoint;
        composeData = np.delete(composeData,minIndex,0);
    
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
    #testWithSyntheticData();
    testKMeans_GroundTruth();
    