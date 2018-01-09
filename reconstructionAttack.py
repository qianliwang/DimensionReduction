import numpy as np;
import matplotlib.pyplot as plot;
import scipy;
from pkg.dimReduction import PCAModule;
from sklearn.cluster import KMeans;
from pkg.svm import SVMModule;
from sklearn.model_selection import ShuffleSplit;
from sklearn.preprocessing import StandardScaler;

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
    
    pcaImpl = PCAModule.PCAImpl(composeData);
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
        augPCAImpl = PCAModule.PCAImpl(augmentedTrainingComData);
        augPCAImpl.getPCs();
        augProjMatrix = augPCAImpl.projMatrix[:,0:numOfPC];
        
        print "Mean vector Distance:";
        print scipy.spatial.distance.cdist([pcaImpl.mean],[augPCAImpl.mean],'cosine');
        
        print "Augmented eigenvector distance:";
        print scipy.spatial.distance.cdist(projMatrix.T,augProjMatrix.T,'cosine').diagonal();
        
    posiTestData = posiData[50:80,:];
    attackPCAImpl = PCAModule.PCAImpl(posiTestData);
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
def testKMeans(path,numOfRounds,varianceRatio,subject):
    data = np.loadtxt(path,delimiter=",");
    rs = ShuffleSplit(n_splits=numOfRounds, test_size=.1, random_state=0);
    rs.get_n_splits(data);
    for train_index, test_index in rs.split(data):

        posiIndices = np.asarray(np.where(data[:,0]==1));
        negIndices = np.asarray(np.where(data[:,0]==-1));
        #print posiIndices;

        scaler = StandardScaler(copy=True);
        scaler.fit(data[train_index,1:]);
        pureTrainingData = scaler.transform(data[train_index,1:]);
        pureTestingData = scaler.transform(data[test_index,1:]);
        print "Training Samples: %d, dimension: %d" % (pureTrainingData.shape[0],pureTrainingData.shape[1]);
        #composeData = data[:,1:];
        pcaImpl = PCAModule.PCAImpl(pureTrainingData);
        pcaImpl.getPCs();
        '''
        print "Original PC's mean vector:";
        print pcaImpl.mean;
        '''
        energies = pcaImpl.getEigValueEnergies();
        numOfPCs = pcaImpl.getNumOfPCwithKPercentVariance(varianceRatio);
        projMatrix = pcaImpl.projMatrix;
        totalEnergy = np.sum(pcaImpl.eigValues);
        print "The total eigenvalue energies is %f, to achieve %f percentage, it needs %d principal components." % (totalEnergy,varianceRatio,numOfPCs);

        numOfCluster = pureTrainingData.shape[0]/pureTrainingData.shape[1];
        print "Maximum number of clusters; %d"% numOfCluster;

        kmeans = KMeans(n_clusters=numOfCluster, random_state=0, n_jobs=5).fit(pureTrainingData);
        '''
        for i in range(len(composeData)):
            print "%d , %d" % (i,kmeans.labels_[i]);
        '''
        gPCAPath = "./"+subject+"_c";
        aPCAPath = "./"+subject+"_appro_c";
        for j in range(2,numOfCluster+1):
            print "%d-means" % j;
            kmeans = KMeans(n_clusters=j, random_state=0, n_jobs=5).fit(pureTrainingData);
            minPCAImpl = None;
            minWeightedDistance = 10000;
            for i in range(j):
                singleClusterData = pureTrainingData[kmeans.labels_ == i];
                print "Cluster %d:%d" % (i,singleClusterData.shape[0]);
                '''
                clusterIndices = np.asarray(np.where(kmeans.labels_ == i));
                #Positive indices intersection and union
                intersectionIndices = np.intersect1d(clusterIndices[0],posiIndices[0],True);
                unionIndices = np.union1d(clusterIndices[0],posiIndices[0]);
                print "Jaccard similarity of positive indices: %f" % (1.0*len(intersectionIndices)/len(unionIndices));
           
                #Negative indices intersection and union
                intersectionIndices = np.intersect1d(clusterIndices[0],negIndices[0],True);
                unionIndices = np.union1d(clusterIndices[0],negIndices[0]);
                print "Jaccard similarity of negative indices: %f" % (1.0*len(intersectionIndices)/len(unionIndices));
                '''
                if singleClusterData.shape[0] >= numOfPCs:
                    tmpPCAImpl = PCAModule.PCAImpl(singleClusterData);
                    tmpPCAImpl.getPCs();
                    #print "Approximate PC's mean vector:";
                    #print tmpPCAImpl.mean;
                    #print "Eigenvalue energies:";
                    #print tmpPCAImpl.getEigValueEnergies();
                    tmpTotalEnergy = np.sum(tmpPCAImpl.eigValues);
                    #print "Total energy of cluster %d data is %f, it takes over %f of the whole data energy." % (i,tmpTotalEnergy,(tmpTotalEnergy/totalEnergy));
                    """
                    Since the PCs contain possible positive and negative values,
                    so the cosine similarity is between -1 and 1,
                    then the cosine distance is 1 - cosine similarity, then it ranges
                    between 0 and 2.
                    """
                    simDist = scipy.spatial.distance.cdist(projMatrix.T[:numOfPCs],tmpPCAImpl.projMatrix.T[:numOfPCs],'cosine');
                    #print simDist.diagonal();
                    rawSimDist = np.sum(simDist.diagonal());
                    #print "Cluster %d Cosine Distance in total: %f" % (i, rawSimDist);
                    '''
                    approSim = 1 - simDist.diagonal();
                    print "Approximate Similarity:"
                    print approSim;
                    '''
                    weightedDistance = 0;
                    for j in range(numOfPCs):
                        weightedDistance = weightedDistance + simDist.diagonal()[j]*energies[j];

                    #print "Weighted Cosine Distance is: %f" % (weightedDistance);
                    optimizationTarget = weightedDistance + (tmpTotalEnergy/totalEnergy);
                    #print "Optimization target: %f" % optimizationTarget;

                    if minWeightedDistance > weightedDistance:
                        minWeightedDistance = weightedDistance;
                        minPCAImpl = tmpPCAImpl;
                        minClusterIndex = i;
                        minRawSimDist = rawSimDist;
                        minOptimizationTarget = optimizationTarget;
                    print "\n";

            print "Minimum cluster index is %d, min raw cosine distance is %f, min weighted cosine distance is %f, min optimization target is %f." % (minClusterIndex,minRawSimDist,minWeightedDistance,minOptimizationTarget);

            for i in range(numOfCluster):
                singleClusterData = pureTrainingData[kmeans.labels_ == i];
                reducedAData = minPCAImpl.transform(singleClusterData,numOfPCs);
                #np.savetxt(aPCAPath+str(i),reducedAData,delimiter=",",fmt='%1.2f');

                reducedGData = pcaImpl.transform(singleClusterData,numOfPCs);
                #np.savetxt(gPCAPath+str(i),reducedGData,delimiter=",",fmt='%1.2f');
                #print pcaImpl.transform(singleClusterData,numOfPCs);


            oriPCAReducedData = pcaImpl.transform(pureTrainingData,numOfPCs);
            approPCAReducedData = tmpPCAImpl.transform(pureTrainingData,numOfPCs);
            #print composeData.shape;

            testOriPCAReducedData = pcaImpl.transform(pureTestingData,numOfPCs);
            testApproPCAReducedData = tmpPCAImpl.transform(pureTestingData,numOfPCs);
            #print testData[:,1:].shape;
            '''
            SVMModule.SVMClf.rbfSVM(oriPCAReducedData,data[:,0],testOriPCAReducedData,testData[:,0]);
            print("=====================================");
            SVMModule.SVMClf.rbfSVM(approPCAReducedData,data[:,0],testApproPCAReducedData,testData[:,0]);
            '''
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
    cumEnergies = 0.95;
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
    #testKMeans_GroundTruth();
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
    path = "./input/"+subject+"_prePCA";

    testKMeans(path,numOfRounds,varianceRatio,subject);
    