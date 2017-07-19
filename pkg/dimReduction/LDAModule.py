import numpy as np;
from numpy import linalg as LA;

class LDAImpl(object):
    
    def __init__(self,scaledData,labels):
        self.numOfClass = 0;
        self.labeledScaleData = np.concatenate((np.reshape(labels,(-1,1)),scaledData),axis=1);
        self.Sw,self.Sb = self.__getSwAndSb(self.labeledScaleData);
        self.SwInv = LA.inv(self.Sw);
        self.projMatrix = None;
        
    def __getLDAPCs(self):
        '''
        Self implemented LDA, the result is same with the implementation of scikit-learn.
        '''
        withinBetweenMatrix = np.dot(self.SwInv,self.Sb);
        w,v = self.eigDecompose(withinBetweenMatrix);
        
        self.projMatrix = v[:,0:self.numOfClass-1];
        
    def getPCs(self):
        self.__getLDAPCs();
        
    def eigDecompose(self,covMatrix):
        w,v = LA.eig(covMatrix);
        idx = np.absolute(w).argsort()[::-1];
        #print idx;
        sortedW = w[idx];
        sortedV = v[:,idx];
        #print sortedW;
        #print sortedV;
        return sortedW,np.real(sortedV);
    
    def __getSwAndSb(self,labeledData):
        '''
        Compute the withinClass covariance and betweenClass covariance.
        '''
        # Using a dictionary to categorize the data.
        labelDict = {};
        for d in labeledData:
            if(d[0] in labelDict):
                labelDict[d[0]].append(d[1:]);
            else:
                tmpList = [];
                tmpList.append(d[1:]);
                labelDict[d[0]] = tmpList;
                
        # Initialize the numOfClass.
        self.numOfClass = len(labelDict.keys());
        
        # Compute the withinClass covariance.
        aggreDict = {};
        tmpTotalMean = None;
        tmpTotalCount = 0;
        withinClassCov = None;
        
        for key,value in labelDict.iteritems():
            tmpNdArray = np.asarray(value);
            if tmpTotalMean is None:
                tmpTotalMean = np.zeros(tmpNdArray.shape[1]);
            if withinClassCov is None:
                withinClassCov = np.zeros((tmpNdArray.shape[1],tmpNdArray.shape[1]));
            withinClassMean = np.mean(tmpNdArray,axis = 0);
            centeredNdArray = tmpNdArray-withinClassMean;
            withinClassCov = withinClassCov + np.dot(centeredNdArray.T,centeredNdArray);
            aggreDict[key] = [withinClassMean,tmpNdArray.shape[0]];
            tmpTotalMean = tmpTotalMean+np.sum(tmpNdArray,axis = 0);
            tmpTotalCount = tmpTotalCount + tmpNdArray.shape[0];
        totalMean = np.divide(tmpTotalMean,tmpTotalCount);
        covSize = len(tmpTotalMean);
        
        # Compute the betweenClass covariance.
        betweenClassCov = np.zeros((covSize,covSize));
        
        for key,value in aggreDict.iteritems():
            withinClassMean = value[0];
            withinClassCount = value[1];
            tmpMeanDiff = withinClassMean - totalMean;
            tmpBetweenCov = np.outer(tmpMeanDiff,tmpMeanDiff.T);
            betweenClassCov = betweenClassCov + withinClassCount*tmpBetweenCov;
        
        return withinClassCov,betweenClassCov;
        
    def transform(self,scaledData,numOfComponents):
        if(self.projMatrix is None):
            self.getPCs();
        if(numOfComponents>(self.numOfClass-1)):
            print "This discrimination could only project data up to %d dimension due the number of classes." % (self.numOfClass-1);
        tmpNumOfComponents = self.numOfClass-1 if numOfComponents>(self.numOfClass-1) else numOfComponents;
        
        tmpProjMatrix = self.projMatrix[:,0:tmpNumOfComponents];
        return np.dot(scaledData,tmpProjMatrix);
