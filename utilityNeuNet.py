import keras;
from keras import optimizers, regularizers;
from keras.models import Sequential;
from keras.layers import Dense,Activation,Dropout;
from keras import backend as K;
from keras.callbacks import Callback, ModelCheckpoint,ReduceLROnPlateau;

import tensorflow as tf;

from pkg.DimReduction import PCAImpl;
from pkg.DPDimReduction import DiffPrivPCAImpl,DiffPrivImpl,DPPro;
from pkg.global_functions import globalFunction as gf;

from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold;
from sklearn.preprocessing import StandardScaler;
from sklearn.metrics import accuracy_score,confusion_matrix, f1_score, precision_score, recall_score;

import numpy as np;
import pandas as pd;
from numpy import linalg as LA;
import sys;
import os;
from time import time;
import matplotlib.pyplot as plt;

###################################
# TensorFlow wizardry
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.95
 
# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))
###################################
'''
class Metrics_callback(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        #val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_predict = self.model.predict_classes(self.validation_data[0]);
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print "--val_f1: %f , val_precision: %f , val_recall %f" % (_val_f1, _val_precision, _val_recall)
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
'''


def drawF1Score(datasetTitle, data=None, path=None, figSavedPath=None):
    plt.clf();
    if path is not None:
        data = np.loadtxt(path, delimiter=",");
    numOfTrails = data.shape[0] / 10;
    print "Number of points on x-axis: %d" % numOfTrails;
    """
    minVector = np.amin(data[:,1:],axis=0);
    yMin = min(minVector);
    maxVector = np.amax(data[:,1:],axis=0);
    yMax = max(maxVector);

    yMin = (yMin-0.1) if (yMin-0.1)>0 else 0;
    yMax = (yMax+0.1) if (yMax+0.1)<1 else 1;
    #x = [10,40,70,100,130,160,190,220,250,280,310,340];
    y1Line,y2Line,y3Line = plt.plot(x, data[:,1], 'bo-', x, data[:,2], 'r^-',x, data[:,3], 'gs-');

    plt.legend([y1Line,y2Line,y3Line], ['PCA', 'Gaussian Noise','Wishart Noise'],loc=4);
    """
    x = np.arange(100,1100,100);
    targetDimension = 9;
    pcaF1 = data[targetDimension::20,3::3];
    dpdpcaF1 = data[targetDimension+1::20,4::3];
    minVector = np.amin(pcaF1, axis=0);
    yMin = min(minVector);
    maxVector = np.amax(dpdpcaF1, axis=0);
    yMax = max(maxVector);

    yMin = (yMin - 0.05) if (yMin - 0.05) > 0 else 0;
    yMax = (yMax + 0.05) if (yMax + 0.05) < 1 else 1.05;

    pcaF1Mean, pcaF1Std = gf.calcMeanandStd(np.asarray(pcaF1));
    pcaF1ErrorLine = plt.errorbar(x, pcaF1Mean, yerr=pcaF1Std, fmt='b', capsize=4);
    pcaF1Line, = plt.plot(x, pcaF1Mean, 'b-')

    dpdpcaF1Mean, dpdpcaF1Std = gf.calcMeanandStd(np.asarray(dpdpcaF1));
    dpdpcaF1ErrorLine = plt.errorbar(x, dpdpcaF1Mean, yerr=dpdpcaF1Std, fmt='r', capsize=4);
    dpdpcaF1Line, = plt.plot(x, dpdpcaF1Mean, 'r-')

    plt.axis([0, x[-1] + 100, yMin, yMax]);
    # plt.axis([0,10,0.4,1.0]);
    plt.legend([pcaF1Line, dpdpcaF1Line], ['PCA', 'DPDPCA'], loc=4);
    plt.xlabel('Number of Epochs', fontsize=18);
    plt.ylabel('F1-Score', fontsize=18);
    plt.title(datasetTitle, fontsize=18);
    plt.xticks(x);

    if figSavedPath is None:
        plt.show();
    else:
        plt.savefig(figSavedPath + "NN_" + datasetTitle + '.pdf', format='pdf', dpi=1000);


def build_MLP(n_hiddenNeurons,n_features,n_classes):

    print "neurons in hidden layer: %d" % n_hiddenNeurons;
    model = Sequential();
    model.add(Dense(n_hiddenNeurons, input_dim=n_features,activation='sigmoid'));
    #model.add(Activation('sigmoid'));
    #model.add(Dropout(DROPOUT));
    model.add(Dense(n_hiddenNeurons/2,activation='sigmoid'));
    #model.add(Activation('relu'));
    #model.add(Dropout(DROPOUT));
    model.add(Dense(n_classes,activation='sigmoid'));
    #model.summary();
    return model;

def fit_MLP(x_train,y_train,x_test,y_test,n_classes,model=None):
    EPOCH = 50;
    BATCH_SIZE = 32;
    LEARNING_RATE = 0.001
    # split into input (X) and output (Y) variables
    #X = dataset[:,0:8]
    #Y = dataset[:,8]

    #y_train = np_utils.to_categorical(y_train,NB_CLASSES);
    #y_test = np_utils.to_categorical(y_test,NB_CLASSES);
    if model is None:
        model = build_MLP(1000, x_train.shape[1], n_classes);
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=LEARNING_RATE),metrics=['accuracy']);

    # good trick to change values in ndArray.
    # convert labels from [1,-1] to [1,0].
    #y_train[y_train < 0]=0;
    #y_test[y_test < 0]=0;

    # Convert class vectors to binary class matrices.
    y_train_oneHot = keras.utils.to_categorical(y_train, n_classes)
    y_test_oneHot = keras.utils.to_categorical(y_test, n_classes)

    # Fit the model
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0000000001)
    callbacks_list = [reduce_lr];
    model.fit(x_train, y_train_oneHot, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks_list,
              validation_data=(x_test, y_test_oneHot));
    # evaluate the model
    scores = model.evaluate(x_test, y_test_oneHot);
    # print("\nCross validation-%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    y_pred = model.predict_classes(x_test);
    acc = accuracy_score(y_test, y_pred);
    print("Neural net accuracy: %.3f" % (acc,));
    ''' 
    expRes =[];
    init_weights = model.get_weights();
    epoches = np.arange(100,1100,100);
    for singleEpoch in epoches:
        skf = StratifiedKFold(n_splits=KFOLD_SPLITS, shuffle=True);
        res = [];
        for index, (train_indices_cv, val_indices_cv) in enumerate(skf.split(x_train, y_train)):
            x_train_cv, x_val_cv = x_train[train_indices_cv], x_train[val_indices_cv]
            y_train_cv, y_val_cv = y_train[train_indices_cv], y_train[val_indices_cv]
            #model = build_MLP(x_train_cv.shape[0],x_train_cv.shape[1],1);
            model.fit(x_train_cv, y_train_cv, epochs=singleEpoch, batch_size=BATCH_SIZE,verbose=0,validation_split = VALIDATION_SPLIT);
            # evaluate the model
            scores = model.evaluate(x_test, y_test);
            res.append(scores[1]);
            #print("\nCross validation-%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            y_pred = model.predict_classes(x_test);
            f1Score = f1_score(y_pred, y_test);
            res.append(f1Score);
            #print("Cross validation: f1 Score: %f, accuracy: %f." % (f1Score,scores[1]));
            model.set_weights(init_weights); 
        resArray = np.asarray(res);
        resArray = np.reshape(resArray,(-1,2));
        avgRes = np.mean(resArray,axis=0);
        print("Avg cross validation: epoch: %d, accuracy: %f, f1 Score: %f" % (singleEpoch, avgRes[0],avgRes[1]));
        expRes.append(singleEpoch);
        expRes.append(avgRes[0]);
        expRes.append(avgRes[1]);
    '''
    return acc;

def singleExp(xEpsilons,trainingData,testingData,largestReducedFeature):
    pureTrainingData = trainingData[:,1:];
    trainingLabel = trainingData[:,0];

    numOfTrainingSamples = trainingData.shape[0];

    pureTestingData = testingData[:,1:];
    testingLabel = testingData[:,0];

    scaler = StandardScaler();
    # print pureTrainingData[0];
    # scaler.fit(pureTrainingData);
    pureTrainingData = scaler.fit_transform(pureTrainingData);
    # print pureTrainingData[0];

    # print pureTestingData[0];
    pureTestingData = scaler.transform(pureTestingData);
    # print pureTestingData[0];

    pcaImpl = PCAImpl(pureTrainingData);
    pcaImpl.getPCs(largestReducedFeature);

    projTrainingData = pcaImpl.transform(pureTrainingData, largestReducedFeature);
    projTestingData = pcaImpl.transform(pureTestingData, largestReducedFeature);
    pcaResult = fit_MLP(projTrainingData, trainingLabel, projTestingData, testingLabel, n_classes=40);

    dpGaussianPCAImpl = DiffPrivPCAImpl(pureTrainingData);
    delta = np.divide(1.0,numOfTrainingSamples);

    # print projTrainingData.shape;
    cprResult = [];
    print "non noise PCA NN training";

    for k, targetEpsilon in np.ndenumerate(xEpsilons):
        #print pcaImpl.projMatrix[:,0];
        #print k;
        print "epsilon: %.2f, delta: %f" % (targetEpsilon, delta);
        cprResult.append(targetEpsilon);
        isGaussianDist = True;
        dpGaussianPCAImpl.setEpsilonAndGamma(targetEpsilon, delta);
        dpGaussianPCAImpl.getDiffPrivPCs(isGaussianDist, largestReducedFeature);

        cprResult.append(pcaResult);

        projTrainingData = dpGaussianPCAImpl.transform(pureTrainingData, largestReducedFeature);
        projTestingData = dpGaussianPCAImpl.transform(pureTestingData, largestReducedFeature);

        result = fit_MLP(projTrainingData,trainingLabel,projTestingData,testingLabel,n_classes=40);
        
        cprResult.append(result);

    cprResult = np.asarray(cprResult);
    return cprResult.reshape((len(xEpsilons), -1));

def doExp(datasetPath,varianceRatio,xEpsilons,n_trials,logPath):
    if os.path.basename(datasetPath).endswith('npy'):
        data = np.load(datasetPath);
    else:
        #data = np.loadtxt(datasetPath, delimiter=",");
        data = pd.read_csv(datasetPath, delimiter=",", header=None).values;
    scaler = StandardScaler();
    data_std = scaler.fit_transform(data[:,1:]);
    globalPCA = PCAImpl(data_std);

    numOfFeature = data.shape[1]-1;
    largestReducedFeature = globalPCA.getNumOfPCwithKPercentVariance(varianceRatio);
    print "%d/%d dimensions captures %.2f variance." % (largestReducedFeature,numOfFeature,varianceRatio);
    cprResult = [];
    rs = StratifiedShuffleSplit(n_splits=n_trials, test_size=.15, random_state=0);
    rs.get_n_splits(data[:,1:],data[:,0]);

    for train_index, test_index in rs.split(data[:,1:],data[:,0]):
        trainingData = data[train_index];
        testingData = data[test_index];
        
        tmpResult = singleExp(xEpsilons, trainingData, testingData, largestReducedFeature);
        with open(logPath, "a") as f:
            np.savetxt(f, tmpResult, delimiter=",", fmt='%1.3f');
        cprResult.append(tmpResult);

    cprResult = np.vstack(cprResult);
    for result in cprResult:
        print ','.join(['%.3f' % num for num in result]);

    return cprResult;


if __name__ == "__main__":
    n_trials = 1;
    logPath = './log/';
    varianceRatio = 0.95;
    xEpsilons = [0.1,0.3,0.5,0.7,0.9]

    if len(sys.argv) > 1:
        datasetPath = sys.argv[1];
        print "+++ using passed in arguments: %s" % (datasetPath);
        result = doExp(datasetPath,varianceRatio,xEpsilons,n_trials,logPath=logPath+"nn_"+os.path.basename(datasetPath)+'.out');
    else:
        #datasets = ['diabetes','Amazon_2','Australian','german','ionosphere'];
        datasets = ['YaleB'];
        for dataset in datasets:
            print "++++++++++++++++++++++++++++  "+dataset+"  +++++++++++++++++++++++++";
            datasetPath = "./input/"+dataset+"_prePCA";
            result = doExp(datasetPath,varianceRatio,xEpsilons,n_trials,logPath=logPath+'nn_'+dataset+'.out');
