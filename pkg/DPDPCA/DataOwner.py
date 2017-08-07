import numpy as np;
import os;
from paillierImpl import *;
from ..diffPriveDimReduction.invwishart import *;
from ..global_functions import globalFunction;
from numpy import linalg as LA;
import math;
import decimal;

class DataOwnerImpl(object):
    
    def __init__(self,dataSharePath):
        self.dataSharePath = dataSharePath;
        self.pub = None;
        tmpData = np.loadtxt(self.dataSharePath,delimiter=",");
        # Here, it should be OK with data[:,1:data.shape[1]];
        self.data = tmpData[:,1:];
    
    def setPub(self,pub):
        self.pub = pub;
        
    def calcAndEncryptDataShare(self):
        
        matrix = self.data;
        
        if not isinstance(matrix,(int, long )):
            matrix = matrix.astype(int);
               
        R = np.dot(matrix.T,matrix);       
        v = np.sum(matrix,axis=0);
        N = matrix.shape[0];
        #matrix = discretize(matrix);
        
        #print "Plaintext R:";
        #print R;
        
        #print R.shape;
        #print v.shape;
        #print N;
        #R = R.astype(int);
        #v = np.floor(v);
        #print "Encrypting R:";
        encR = np.empty((R.shape[0],R.shape[1]),dtype=np.dtype(decimal.Decimal));
        for i in range(0,R.shape[0]):
            for j in range(0,R.shape[1]):
                encR[i,j] = encrypt(self.pub,R[i,j]);
        '''        
        it = np.nditer(R, flags=['multi_index']);
        while not it.finished:
            #print "%d <%s>" % (it[0], it.multi_index),
            #print str(it.multi_index)+","+str(it[0]);
            encR[it.multi_index] = encrypt(pub,it[0]);
            it.iternext();
        '''
        #print "Encrypt v:";
        encV = np.empty(v.shape[0],dtype=np.dtype(decimal.Decimal));
        for i in range(0,len(v)):
            encV[i] = encrypt(self.pub,v[i]);
        '''
        it = np.nditer(v, flags=['multi_index'])
        while not it.finished:
            #print "%d <%s>" % (it[0], it.multi_index),
            encV[it.multi_index] = encrypt(pub,it[0]);
            it.iternext();
        '''
        encN = encrypt(self.pub,N);
        return encR,encV,(encN,);
    
    def discretize(self,R):
        #e is the number of bits, not the natural number.
        e = 8;
        cMax = np.amax(R,axis=0);
        #print cMax;
        cMin = np.amin(R,axis=0);
        #print cMin;
        intR = np.floor(np.divide((math.pow(2,e)-1)*(R - cMin),(cMax-cMin)));
        return intR;
    
    def __saveEncrypedData(self,outputPath,matrix):
        np.savetxt(outputPath,matrix,delimiter=",",fmt="%0x");
        
    def saveShares(self,encFolderPath,fileName,encR,encV,encN):
        
        encRFilePath = encFolderPath+"encR/"+fileName;
        encVFilePath = encFolderPath+"encV/"+fileName;
        encNFilePath = encFolderPath+"encN/"+fileName;
        
        if not os.path.exists(encFolderPath+"encR/"):
            os.system('mkdir -p %s' % (encFolderPath+"encR/"));
            os.system('mkdir -p %s' % (encFolderPath+"encV/"));
            os.system('mkdir -p %s' % (encFolderPath+"encN/"));
        
        self.__saveEncrypedData(encRFilePath,encR);
        self.__saveEncrypedData(encVFilePath,encV);
        self.__saveEncrypedData(encNFilePath,encN);
        
        '''
        np.savetxt(encRFilePath,encR,delimiter=",");
        np.savetxt(encVFilePath,encV,delimiter=",");
        np.savetxt(encNFilePath,np.array(encN).reshape(1,),delimiter=",");
        '''
        
    def privateLocalPCA(self,data,k,epsilon):
        
        if data is None:
            data = self.data;
            
        k = np.minimum(k,LA.matrix_rank(data));
        #print "In each data owner, the k is: %d" % k;
            
        C = np.dot(data.T,data);
        if epsilon is not 0:
            df = len(C)+1;
            sigma = 1/epsilon*np.identity(len(C));
            #print sigma;
            wishart = wishartrand(df,sigma);
            U, s, V = np.linalg.svd(C+wishart);
        else:
            U, s, V = np.linalg.svd(C);
        #U, s, V = LA.svd(C);
        S = np.diagflat(np.sqrt(s));
    #    print U[:,0:k].shape;
    #    print S[0:k,0:k].shape;
        P = np.dot(U[:,0:k],S[0:k,0:k]);
        #sqrtS = np.sqrt(s);
        #print sqrtS;
        #tmpSum = np.sum(sqrtS);
        #print [elem/tmpSum for elem in sqrtS];
        return P;
'''    
dataSharePath = "./input/australian_prePCA_referPaper/0";
R,v,N = calcDataShare(dataSharePath);
priv, pub = generate_keypair(128);
print "public key is "+ str(pub);
encR,encV,encN = encryptDataShare(R,v,N,pub);
encFolderPath = dataSharePath+"_enc/";
saveShares(encFolderPath,encR,encV,encN);
'''

