import datetime;
import pandas as pd;
import numpy as np;

def getCSVFromArff(fileNameArff,fileNameCsv):

    with open(fileNameArff + '.arff', 'r') as fin:
        data = fin.read().splitlines(True)
    
    i = 0
    cols = []
    for line in data:
        if ('@data' in line):
            i+= 1
            break
        else:
            #print line
            i+= 1
            if (line.startswith('@attribute')):
                if('{' in line):
                    cols.append(line[11:line.index('{')-1])
                else:
                    cols.append(line[11:line.index('numeric')-1])
    
    headers = ",".join(cols)
    
    with open(fileNameCsv + '.csv', 'w') as fout:
        #fout.write(headers)
        #fout.write('\n')
        fout.writelines(data[i:])

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def alldates(values,f):
    for v in values:
        if(not is_date(v,f)):
            return False
    
    return True

def is_date(s,f):
    try:
        # example f : '%Y-%m-%d'
        datetime.datetime.strptime(s, f)
        return True
    except ValueError:
        return False
        
def getArffFile(fullDataSetFileName,trainFileName,target_nominal_column_name,colsExclude=[],maxNonimalCardinality=100,dateformat='%Y-%m-%d'):
    
    # Here fullDataSetFileName is the full dataset while trainFileName is a part used for training or testing
    # The reason you need fullDataSetFileName, is you need to make sure you have all the unique values for a feature
    # 
    # The train file being a subset, might not have all the unique values of a feature and hence
    # while using the same for creating a test file, you might get different values. In such cases Weka
    # will not let you test on that set.
    
    # If you do not need this functionality, such remove fullDataSetFileName from the arguments 
    # and work only with the trainFileName

    # target_nominal_column_name is the column which will generally be the last
    # feature in the arff while, which you want to predict
    
    # colsExclude is a list of columns you want to exclude from the arff file
    # pass an [] list if you want them all
    
    # I also added a maxNonimalCardinality flag. 
    # This is used to make sure that numeric values can also be considered as nominal
    # if there are very few of them
    
    
    # read full data set
    dfResourceFull = pd.read_csv(fullDataSetFileName)
    
    # read train/test data set and temporily same a csv verion of only included columns    
    dfResourceTrain = pd.read_csv(trainFileName)
    colsInclude = [col for col in dfResourceTrain.columns if col not in colsExclude]
    
    dfResourceTrain = dfResourceTrain[colsInclude]
    dfResourceTrain.to_csv('tmp.csv',index=False)

    # read the temporary csv file    
    with open('tmp.csv', 'r') as fin:
        data = fin.read().splitlines(True)
    
    # first line in arff
    output =  '@relation ' + fullDataSetFileName + '\n'
    #print
    for col in colsInclude:
        unqValues = pd.Series(dfResourceFull[col].values.ravel()).unique()
        # check type of values
        
        # use date if all dates
        if( str(dfResourceTrain.dtypes[col]).startswith('datetime64') or (str(dfResourceTrain.dtypes[col]) == 'object' and alldates(unqValues,dateformat))):
            x =  '@attribute' + ' ' + col + ' date "' + dateformat + '"'
            output =  output + '\n' + x

        # using string, when there are strings and the cardinality is high
        elif( (len(unqValues) > maxNonimalCardinality) and (str(dfResourceTrain.dtypes[col]) == 'object')):
            x =  '@attribute' + ' ' + col + ' string'
            output =  output + '\n' + x
        
        # use numeric if numeric and the cardinality is high
        elif( (len(unqValues) > maxNonimalCardinality) and ((str(dfResourceTrain.dtypes[col]) == 'float64') or (str(dfResourceTrain.dtypes[col]) == 'int64'))):
            x =  '@attribute' + ' ' + col + ' numeric'
            output =  output + '\n' + x
        
        # use nominal
        else:
            x = ''
            for s in unqValues:
                x = x + str(s) + ','
            x =  '@attribute' + ' ' + col + ' {' + x + '}'
            x = x.replace(',}', '}')
            #print x
            output =  output + '\n' + x


    output =  output + '\n' 
    #print '@data'
    output =  output + '\n' + '@data' + '\n'

    with open(trainFileName+'.arff', 'w') as fout:
        fout.write(output)
        fout.writelines(data[1:])
        
if __name__ == "__main__":
    arffPath = "/Users/Watson/Development/Dataset/Text/Amazon_initial_50_30_10000";
    csvPath = "/Users/Watson/Development/Dataset/Text/Amazon_initial_50_30_10000";
    #getCSVFromArff(arffPath,csvPath);
    names = ['Agresti','Ashbacher','Auken','Blankenship','Brody','Brown','Bukowsky','CFH','Calvinnme','Chachra','Chandler','Chell','Cholette','Comdet','Corn','Cutey','Davisson','Dent','Engineer','Goonan','Grove','Harp','Hayes','Janson','Johnson','Koenig','Kolln','Lawyeraau','Lee','Lovitt','Mahlers2nd','Mark','McKee','Merritt','Messick','Mitchell','Morrison','Neal','Nigam','Peterson','Power','Riley','Robert','Shea','Sherwin','Taylor','Vernon','Vision','Walters','Wilson'];
    nameDict = {};
    for i,name in enumerate(names):
        nameDict[name] = i+1;
    #print nameDict;
    #print len(nameDict);
    resultArray = np.zeros((1,10001));
    csvClassPath = csvPath+"_class0";
    with open(csvPath+".csv") as f:
        lines=f.readlines();
        for line in lines:
            singleLine = line.split(',');
            nameId = nameDict[singleLine[-1].strip()];
            myarray = np.fromstring(line,sep=',')
            newArray = np.insert(myarray,0,nameId);
            print newArray.shape;
            resultArray = np.append(resultArray,newArray.reshape((1,10001)),axis=0);
    print resultArray[:,0];
    np.savetxt(csvClassPath,resultArray[1:,:],delimiter=",",fmt='%d');
    """
    data = np.genfromtxt(csvPath+".csv",dtype = None, delimiter=",");
    for singleData in data:
        print singleData;
        np.insert(data,0,nameDict[singleData[10000]]);
    print data.shape;
    """