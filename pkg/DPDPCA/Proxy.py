import os;
import numpy as np;
import glob;
import copy;
from paillierImpl import *;
from multiprocessing import Pool;
import time;
import decimal;

'''
def aggrtDataOwnerShares(folderPath):
    dataFileList = glob.glob(folderPath+"/*");
    R,v,N = dataOwnerShare.calcDataShare(dataFileList[0]);
    SumR = copy.copy(R);
    SumV = copy.copy(v);
    SumN = copy.copy(N);
    dataFileList.pop(0);
    for path in dataFileList:
        R,v,N = dataOwnerShare.calcDataShare(path);
        SumR = SumR + R;
        SumV = SumV + v;
        SumN = SumN + N;
    return SumR,SumV,SumN;

def calcEigenvectors(SumR,SumV,SumN):
    totalScatterMatrix = SumR - np.divide(np.dot(SumV,SumV.T),SumN);
    w,v = LA.eig(totalScatterMatrix);
    print w.shape;
    print v.shape;

def aggrtEncryptedDataOwnerShare(self,encFolderPath,pub):
    #print "Aggregating encrypted data shares.."
    print str(int(round(time.time() * 1000)))+", Proxy aggregation start...";
    
    encRList = glob.glob(encFolderPath+"encR/*");
    encVList = glob.glob(encFolderPath+"encV/*");
    encNList = glob.glob(encFolderPath+"encN/*");
    
    sumEncR = self.__aggrtEncryptedData(encRList,pub);
    sumEncV = self.__aggrtEncryptedData(encVList,pub);
    sumEncN = self.__aggrtEncryptedData(encNList,pub);
    
    print str(int(round(time.time() * 1000)))+", Proxy aggregation ends...";
    
    return sumEncR, sumEncV,sumEncN;
'''

def aggrtEncryptedData(encFileList,pub):
    enc = loadEncryptedData(encFileList[0]);
    
    sumEnc = copy.copy(enc);
    encFileList.pop(0);
    for path in encFileList:
        #print path;
        enc = loadEncryptedData(path);
        sumEnc = addEncryptedData(sumEnc,enc,pub);
         
    return sumEnc;

def addEncryptedData(sumEnc,enc,pub):
    
    if enc.ndim == 2:
        for i in range(0,enc.shape[0]):
            for j in range(0,enc.shape[1]):
                sumEnc[i,j] = e_add(pub, sumEnc[i,j], enc[i,j]);
    elif enc.ndim == 1:
        for i in range(0,enc.shape[0]):
            sumEnc[i] = e_add(pub, sumEnc[i], enc[i]);
    else:
        sumEnc = e_add(pub, sumEnc, enc);
    return sumEnc;
    
def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]
'''
Multi-processing, not mutli-threading
'''
def paraTask(fileList,pub):
    p = Pool(8);
    results = [];
    #numOfTrunks = int(len(fileList)/math.log(len(fileList)));
    numOfTrunks = 4;
    #print "num of trunks:" + str(numOfTrunks);
    subEncSets = split_list(fileList,numOfTrunks);
    # using apply_async
    
    for l in subEncSets:
        results.append(p.apply_async(aggrtEncryptedData, (l,pub)));

    # wait for results
    sumEnc = results[0].get();
    if isinstance(sumEnc,np.ndarray):
        for res in results[1:]:
            addEncryptedData(sumEnc,res.get(),pub);
        #print sumEnc.shape;
    else:
        for res in results[1:]:
            sumEnc = e_add(pub, sumEnc, res.get());
    
    p.close();
    return sumEnc;
    
def paraAggrtEncryptedData(encFolderPath,pub):
    
    print str(int(round(time.time() * 1000)))+", Proxy parallel aggregation start...";
    encRList = glob.glob(encFolderPath+"encR/*");
    sumEncR = paraTask(encRList,pub);
    encVList = glob.glob(encFolderPath+"encV/*");
    sumEncV = paraTask(encVList,pub);
    encNList = glob.glob(encFolderPath+"encN/*");
    sumEncN = paraTask(encNList,pub);
    print str(int(round(time.time() * 1000)))+", Proxy parallel aggregation end...";
    
    return sumEncR,sumEncV,sumEncN;

def initFolders(encFolderPath):
    encRFolderPath = encFolderPath+"encR";
    encVFolderPath = encFolderPath+"encV";
    encNFolderPath = encFolderPath+"encN";
    
    if not os.path.exists(encFolderPath+"encR"):
        os.system('mkdir -p %s' % encRFolderPath);
        os.system('mkdir -p %s' % encVFolderPath);
        os.system('mkdir -p %s' % encNFolderPath);      
    
    return encRFolderPath,encVFolderPath,encNFolderPath;

def loadEncryptedData(inputPath):
    enc = np.loadtxt(inputPath,delimiter=",",dtype='str');
    #strEnc = np.array_str(enc);
    if enc.ndim == 2:
        x = np.empty((enc.shape[0],enc.shape[1]),dtype=np.dtype(decimal.Decimal));
        for i in range(0,enc.shape[0]):
            for j in range(0,enc.shape[1]):
                x[i,j] = int(str(enc[i,j]),16);
    elif enc.ndim == 1:
        x = np.empty(enc.shape[0],dtype=np.dtype(decimal.Decimal));
        for i in range(0,enc.shape[0]):
            x[i] = int(str(enc[i]),16);
    else:
        enc = np.array(int(str(enc),16));
        return enc;
    '''                     
    it = np.nditer(enc, flags=['multi_index']);
    while not it.finished:
        x[it.multi_index] = int(str(it[0]),16);
        it.iternext();
    '''
    return x;
'''
folderPath = "./input/australian_prePCA_referPaper/plaintext/";
#SumR,SumV,SumN = aggrtDataOwnerShares(folderPath);
#calcEigenvectors(SumR,SumV,SumN);
priv, pub = generate_keypair(128);

encFolderPath = "./input/australian_prePCA_referPaper/ciphertext/";
encRFolderPath,encVFolderPath,encNFolderPath = initFolders(encFolderPath);
    
# 1) Data Owner do the computation and encryption.
#callDataOwners(folderPath,encFolderPath,pub);

# 2) The Proxy aggregates the encrypted data shares from each data owner.
sumEncR, sumEncV, sumEncN = aggrtEncryptedDataOwnerShare(encRFolderPath,encVFolderPath,encNFolderPath,pub);

# 3) Data User performs the decryption and the EVD. 
v = dataUserJob(sumEncR,sumEncV,sumEncN,priv,pub);
print v;
'''