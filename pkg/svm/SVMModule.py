from sklearn.model_selection import StratifiedKFold;
from sklearn.model_selection import GridSearchCV;
from sklearn import svm;
from sklearn.metrics import f1_score;
from sklearn.metrics import recall_score;
from sklearn.metrics import precision_score;
import numpy as np;

class SVMClf(object):
    
    @classmethod
    def linearSVM(cls,trainingData,trainingLabel,testingData,testingLabel):
        svmClf = svm.LinearSVC();
        svmClf.fit(trainingData,trainingLabel);
        svmPred = svmClf.predict(testingData);
        #f1Score = f1_score(testingLabel,svmPred,average='weighted');
        result = SVMClf.calcF1Score(testingLabel,svmPred);
        #print "F1Score: %f" % result[2];
        #print result;
        #print "The linear SVM has a f1 score: %f." % f1Score;
        return result;
        
    @classmethod
    def rbfSVM(cls,trainingData,trainingLabel,testingData,testingLabel):
        '''
        The SVM with rbf kernel, Scikit-learn SVM, with grid search Cross Validation.
        '''
        # 1). 10-fold cross validation. 
        skfCV = StratifiedKFold(n_splits=10,shuffle=True);
        
        # 2). Grid search, with the C and gamma parameters.
        C_range = np.logspace(-5, 15, 10,base = 10.0);
        #print C_range;
        gamma_range = np.logspace(-15, 3, 9, base=10.0);
        #print gamma_range;
        param_grid = dict(gamma=gamma_range, C=C_range);
        # Notice here that the svm.SVC is just for searching for the parameter, we didn't really train the model yet.  
        grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, scoring="f1", n_jobs =2, cv=skfCV);
        #grid.fit(ldaProjTrainingData, trainingLabel);
        grid.fit(trainingData, trainingLabel);
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_));
        
        # 3). Train the SVM model using the parameters from grid search. 
        svmClf = svm.SVC(C = grid.best_params_['C'], gamma = grid.best_params_['gamma'], kernel='rbf');
        svmClf.fit(trainingData,trainingLabel);
        accuracy_train = svmClf.score(trainingData,trainingLabel);
        print "Training Score: %f" % accuracy_train; 
        #print svmClf.support_;
        # 4). Predict the testing data and calculate the f1 score.
        svmPred = svmClf.predict(testingData);
        result = SVMClf.calcF1Score(testingLabel,svmPred);
        return result;
    
    @classmethod
    def calcF1Score(cls,testingLabels,predLabels):
        TP = 0;
        TN = 0;
        FP = 0;
        FN = 0;
        for i in range(0,len(testingLabels)):
            if(testingLabels[i]==1):
                if(predLabels[i]==testingLabels[i]):
                    TP = TP + 1;
                else:
                    FN = FN + 1; 
            else:
                if(predLabels[i]==testingLabels[i]):
                    TN = TN + 1;
                else:
                    FP = FP + 1;
        precision = 1.0*TP/(TP+FP) if (TP+FP)!=0 else 0;
        print "TruePositive: %d "% TP;
        print "FalsePositive: %d "% FP;
        print "Precision: %f" % precision;
        recall = 1.0*TP/(TP+FN) if (TP+FN)!=0 else 0;
        print "Recall: %f" % recall;
        F1Score = 2.0*precision*recall/(precision+recall) if (precision+recall)!=0 else 0 ;
        print "F1Score: %f" % F1Score;
        return precision,recall,F1Score;
    