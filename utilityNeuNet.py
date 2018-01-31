from keras.models import Sequential;
from keras.layers import Dense,Activation,Dropout;
from keras.utils import np_utils;
from keras import backend as K;

from pkg.dimReduction import PCAModule;
from pkg.diffPrivDimReduction import DPModule;
from pkg.diffPrivDimReduction import DiffPrivPCAModule;

from sklearn.model_selection import StratifiedShuffleSplit;
from sklearn.model_selection import StratifiedKFold;
from sklearn import preprocessing;
from sklearn.preprocessing import StandardScaler;
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score;

import numpy as np;
from numpy import linalg as LA;
import sys;
import os;

class Metrics_callback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print “ — val_f1: %f — val_precision: %f — val_recall %f” %(_val_f1, _val_precision, _val_recall)
        return;

myCallback = Metrics_callback();

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall));

def create_MLP():
    NUM_FEATURE = x_train.shape[1];
    N_HIDDEN = NUM_FEATURE;
    model = Sequential();
    model.add(Dense(N_HIDDEN, input_dim=NUM_FEATURE,activation='sigmoid'));
    #model.add(Activation('sigmoid'));
    #model.add(Dropout(DROPOUT));
    #model.add(Dense(N_HIDDEN,activation='relu'));
    #model.add(Activation('relu'));
    #model.add(Dropout(DROPOUT));
    model.add(Dense(1,activation='sigmoid'));
    #model.summary();
    # Compile model
    #model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=[f1])
    return model;

def fit_MLP(x_train,y_train,x_test,y_test):
    EPOCH = 100;
    BATCH_SIZE = 100;
    NUM_FEATURE = x_train.shape[1];
    N_HIDDEN = NUM_FEATURE;
    VALIDATION_SPLIT = 0.1;
    #RESHAPED = 856;
    NB_CLASSES = 2;
    DROPOUT = 0.1;
    KFOLD_SPLITS = 10;
    # load pima indians dataset
    #dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    #X = dataset[:,0:8]
    #Y = dataset[:,8]

    #y_train = np_utils.to_categorical(y_train,NB_CLASSES);
    #y_test = np_utils.to_categorical(y_test,NB_CLASSES);

    # good trick to change values in ndArray.
    y_train[y_train < 0]=0;
    y_test[y_test < 0]=0;
    # create model
    model = Sequential();

    model.add(Dense(N_HIDDEN, input_dim=NUM_FEATURE,activation='sigmoid'));
    #model.add(Activation('sigmoid'));
    #model.add(Dropout(DROPOUT));
    #model.add(Dense(N_HIDDEN,activation='relu'));
    #model.add(Activation('relu'));
    #model.add(Dropout(DROPOUT));
    model.add(Dense(1,activation='sigmoid'));
    #model.summary();
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=[f1])
    # Fit the model

    skf = StratifiedKFold(n_splits=KFOLD_SPLITS, shuffle=True);
    
    for index, (train_indices_cv, val_indices_cv) in enumerate(skf.split(x_train, y_train)):
        x_train_cv, x_val_cv = x_train[train_indices_cv], x_train[val_indices_cv]
        y_train_cv, y_val_cv = y_train[train_indices_cv], y_train[val_indices_cv]
        model.fit(x_train_cv, y_train_cv, epochs=EPOCH, batch_size=BATCH_SIZE,verbose=0,validation_split = VALIDATION_SPLIT,callbacks=[myCallback]);
        # evaluate the model
        scores = model.evaluate(x_test, y_test)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return scores[1];

def singleExp(xDimensions,trainingData,testingData,largestReducedFeature,epsilon):
    pureTrainingData = trainingData[:,1:];
    trainingLabel = trainingData[:,0];
    
    pureTestingData = testingData[:,1:];
    testingLabel = testingData[:,0];

    scaler = StandardScaler(copy=False);
    #print pureTrainingData[0];
    scaler.fit_transform(pureTrainingData);
    #print pureTrainingData[0];

    #print pureTestingData[0];
    scaler.transform(pureTestingData);
    #print pureTestingData[0];

    cprResult = [];
    pcaImpl = PCAModule.PCAImpl(pureTrainingData);
    
    pcaImpl.getPCs(largestReducedFeature);
    numOfTrainingSamples = trainingData.shape[0];
    
    delta = np.divide(1.0,numOfTrainingSamples);
    print "epsilon: %.2f, delta: %f" % (epsilon,delta);
    
    isGaussianDist = True;
    dpGaussianPCAImpl = DiffPrivPCAModule.DiffPrivPCAImpl(pureTrainingData);
    dpGaussianPCAImpl.setEpsilonAndGamma(epsilon,delta);
    dpGaussianPCAImpl.getDiffPrivPCs(isGaussianDist,largestReducedFeature);
    
    for k, targetDimension in np.ndenumerate(xDimensions):    
        #print pcaImpl.projMatrix[:,0];
        #print k;
        cprResult.append(targetDimension);
        projTrainingData1 = pcaImpl.transform(pureTrainingData,targetDimension);
        projTestingData1 = pcaImpl.transform(pureTestingData,targetDimension);
        print "Non-noise PCA %d" % targetDimension;
        result = MLP(projTrainingData1,trainingLabel,projTestingData1,testingLabel);
        
        cprResult.append(result);

        projTrainingData2 = dpGaussianPCAImpl.transform(pureTrainingData,targetDimension);
        projTestingData2 = dpGaussianPCAImpl.transform(pureTestingData,targetDimension);
        print "Gaussian-noise PCA %d" % targetDimension;
        
        result = fit_MLP(projTrainingData2,trainingLabel,projTestingData2,testingLabel);
        
        cprResult.append(result);
        """
        projTrainingData3 = dpWishartPCAImpl.transform(pureTrainingData,targetDimension);
        projTestingData3 = dpWishartPCAImpl.transform(pureTestingData,targetDimension);
        print "Wishart-noise PCA %d" % targetDimension;
        if isLinearSVM:
            result = SVMModule.SVMClf.linearSVM(projTrainingData3,trainingLabel,projTestingData3,testingLabel);
        else:
            result = SVMModule.SVMClf.rbfSVM(projTrainingData3,trainingLabel,projTestingData3,testingLabel);
        cprResult.append(result[3]);
        """

    resultArray = np.asarray(cprResult);
    resultArray = np.reshape(resultArray, (len(xDimensions), -1));
    return resultArray;

def doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfDimensions):
    if os.path.basename(datasetPath).endswith('npy'):
        data = np.load(datasetPath);
    else:
        data = np.loadtxt(datasetPath, delimiter=",");
    scaler = StandardScaler();
    data_std = scaler.fit_transform(data[:,1:]);
    globalPCA = PCAModule.PCAImpl(data_std);

    numOfFeature = data.shape[1]-1;
    largestReducedFeature = globalPCA.getNumOfPCwithKPercentVariance(varianceRatio);
    print "%d/%d dimensions captures %.2f variance." % (largestReducedFeature,numOfFeature,varianceRatio);
    xDimensions = None;
    if numOfDimensions > numOfFeature:
        xDimensions = np.arange(1,numOfFeature);
        largestReducedFeature=numOfFeature;
    else:
        xDimensions = np.arange(1,largestReducedFeature,max(largestReducedFeature/numOfDimensions,1));
    
    cprResult = None;
    rs = StratifiedShuffleSplit(n_splits=numOfRounds, test_size=.2, random_state=0);
    rs.get_n_splits(data[:,1:],data[:,0]);

    for train_index, test_index in rs.split(data[:,1:],data[:,0]):
        trainingData = data[train_index];
        testingData = data[test_index];
        
        tmpResult = singleExp(xDimensions, trainingData, testingData, largestReducedFeature, epsilon);
        if cprResult is None:
            cprResult = tmpResult;
        else:
            cprResult = np.concatenate((cprResult,tmpResult),axis=0);


    for result in cprResult:
        print ','.join(['%.3f' % num for num in result]);

    return cprResult;
if __name__ == "__main__":
    numOfRounds = 4;
    resultSavedPath = "./log/";
    numOfDimensions = 15;
    epsilon = 0.3;
    varianceRatio = 0.9;
    
    if len(sys.argv) > 1:
        datasetPath = sys.argv[1];
        print "+++ using passed in arguments: %s" % (datasetPath);
        result = doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfDimensions);
        np.savetxt(resultSavedPath+"numPC_NN_"+os.path.basename(datasetPath)+".output",result,delimiter=",",fmt='%1.3f');
    else:
        datasets = ['diabetes','CNAE_2','CNAE_5','CNAE_7','face2','Amazon_3','madelon'];
        #datasets = ['diabetes','Amazon_2','Australian','german','ionosphere'];
        for dataset in datasets:
            print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
            datasetPath = "./input/"+dataset+"_prePCA";
            result = doExp(datasetPath,epsilon,varianceRatio,numOfRounds,numOfDimensions);
            np.savetxt(resultSavedPath+"numPC_NN_"+dataset+".output",result,delimiter=",",fmt='%1.3f');
