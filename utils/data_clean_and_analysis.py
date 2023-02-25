import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def remove_id():
    """do not use this function """
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    train.to_csv('data/train_no_id.csv')
    test.to_csv('data/test_no_id.csv')

def get_data():
     train = pd.read_csv('data/train_no_id.csv').iloc[:,1:]
     test = pd.read_csv('data/test_no_id.csv').iloc[:,1:]
     augmented = pd.read_csv('data/augmented_test_no_id.csv')
     return train,test,augmented

def histogram(data:pd.DataFrame,save = False):
    """draw the histogram of input data
        histogram is drawn based on columns 
    """

    for i in range(len(data.columns)):
        plt.hist(data.iloc[:,i],bins = 100)
        plt.title("histogram of "+data.columns[i])
        if save == True:
            plt.savefig('figure/histogram_'+str(data.columns[i]))
        plt.show()

def heatmap(data:pd.DataFrame,save = False):
    """ draw the heatmap of given df"""
    sns.heatmap(data.corr())
    if save == True:
            plt.savefig('figure/heatmap')
    plt.show()

def minmax(data:pd.DataFrame,label = False):
    """ do the minmax normalization based on whether data has y
        by default we assume that the label is in the last line
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
         data_minmax['labl'] = y.values
    return data_minmax

def result_comparition(data:np.ndarray):
    """
    here we input the predicted results and compard with the file submission.csv
    score is measured by the number of different predictions
    """
    a = pd.read_csv("output_record/submission.csv")['prediction'].values
    score = np.sum(np.abs(a-data))
    score_rate = score/len(data)
    return score, score_rate