import numpy as np;
from numpy import linalg as LA;
from ..paillier import paillierImpl;
import time;

class DataUserImpl(object):
    
    def __init__(self,pub,priv):
        
        self.pub = pub;
        self.priv = priv;
    
    def decryptAndEVD(self,encR,encV,encN,priv,pub):
        
        print str(int(round(time.time() * 1000)))+", Data User decryption starts.."
        
        # Decrypt encR
        R = np.zeros((encR.shape[0],encR.shape[1]));
        
        for i in range(0,encR.shape[0]):
            for j in range(0,encR.shape[1]):
                R[i,j] = paillierImpl.decrypt(priv,pub,int(encR[i,j]));
        #print "Decrypted Aggregated R:";
        #print R;
        
        # Decrypt encV
        v = np.zeros(encV.shape[0]); 
        for i in range(0,encV.shape[0]):
            v[i] = paillierImpl.decrypt(priv,pub,int(encV[i]));
        #print "Decrypted Aggregated V:";
        #print v;
        
        # Decrypt encN
        N = paillierImpl.decrypt(priv,pub,int(encN));
        
        # Performing EVD on decrypted result.
        aggr = R - np.divide(np.dot(v,v.T),N);
        w, v = LA.eig(aggr);
        
        print str(int(round(time.time() * 1000)))+", Data User decryption ends.."
        
        return v;
