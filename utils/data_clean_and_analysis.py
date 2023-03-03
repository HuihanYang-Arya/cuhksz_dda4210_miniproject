import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split

def remove_id():
    """do not use this function """
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    train.to_csv('data/train_no_id.csv')
    test.to_csv('data/test_no_id.csv')

def get_data():
     train = pd.read_csv('data/train_no_id.csv').iloc[:,1:]
     test = pd.read_csv('data/test_no_id.csv').iloc[:,1:]
     augmented = pd.read_csv('data/augmented_test_no_id.csv').iloc[:,1:]
     return train,test,augmented

def histogram(data:pd.DataFrame,save = False):
    """draw the histogram of input data
        histogram is drawn based on columns and features
    """
    for i in range(len(data.columns)):
        plt.hist(data.iloc[:,i],bins = 100)
        plt.title("histogram of "+data.columns[i])
        if save == True:
            plt.savefig('figure/histogram_'+str(data.columns[i]))
        plt.show()

def heatmap(data:pd.DataFrame,save = False):
    """ draw the heatmap of given data"""
    sns.heatmap(data.corr(),cmap='summer')
    if save == True:
            plt.savefig('figure/heatmap')
    plt.show()

def minmax(data:pd.DataFrame,label = False):
    """ do the minmax normalization 
        data:       Data need to be normalized
        label:      Whether data is labeled. If it is labeled, we assume label column is in the last column.
    """
    if label == True:
          y = data.iloc[:,-1]
          data = data.iloc[:,:-1]
    scaler = MinMaxScaler()
    data_minmax = pd.DataFrame(columns=data.columns)
    for i in range(len(data.columns)):
        scaler.fit(data.iloc[:,i].values.reshape(-1,1))
        data_minmax.iloc[:,i] = scaler.transform(data.iloc[:,i].values.reshape(-1,1)).reshape(1,-1)[0]
    if label == True:
         data_minmax['lable'] = y.values
    return data_minmax

def result_comparition(data:np.ndarray):
    """
    here we input the predicted results and compard with the file submission.csv
    score is measured by the number of different predictions
    -not important for the augmented data
    """
    a = pd.read_csv("output_record/submission.csv")['prediction'].values
    score = np.sum(np.abs(a-data))
    score_rate = score/len(data)
    return score, score_rate

def outlier_remove(data:pd.DataFrame,label = False,n_neighbors = 2):
    """
    function used to remove the outlier
    """
    if label == True:
          y = data.iloc[:,-1]
          data = data.iloc[:,:-1]
    print(n_neighbors)
    for i in range(len(data.columns)):
         clf = LocalOutlierFactor(n_neighbors)
         outlier = clf.fit_predict(data.iloc[:,i].values.reshape(-1,1))
         print(np.sum(outlier == -1))
    
    
def split_normal(train_data:pd.DataFrame,test_data:pd.DataFrame,split_size = 0.8,normal = True):
    """
    Function used to split the train data into training set and validation set and perform min-max normalization.
    train_data:         training dataset with label
    test_data:          test dataset with label
    split_size(float):  the partition of training set(split_size) and validation set(1-split_size).
    """
    train_tr_data,val_data = train_test_split(train_data,test_size=(1-split_size),train_size=split_size)
    train_y = train_tr_data.iloc[:,-1]
    train_x = train_tr_data.iloc[:,:-1]
    val_y = val_data.iloc[:,-1]
    val_x = val_data.iloc[:,:-1]
    if normal == True:
        train_x = minmax(train_x)
        val_x = minmax(val_x)
        test_data = minmax(test_data)
    return train_x,train_y,val_x,val_y,test_data

def test1(function):
    """
    function used to do the training and validation test
    """
    train,test,argu = get_data()
    result = []
    for i in range(40):
        result.append(function(train,test)*100)
    return np.mean(result),np.var(result)

def test_aug(function,name = 'output_record/tmp.csv'):
    """
    function used to get the statistic of the performance on augmented dataset.
    """
    train,test,argu = get_data()
    function(train,argu,train = False,save = name)
    return count(name = name)

def count(target = 1,name = 'output_record/tmp.csv'):
    df = pd.read_csv(name).iloc[:,1].values
    return np.count_nonzero(df == target)
