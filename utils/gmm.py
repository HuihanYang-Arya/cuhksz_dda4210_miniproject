import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from utils import data_clean_and_analysis
from sklearn.ensemble import AdaBoostClassifier

def i_gmm(train_data, test_data, normal = True, save = "output_record/tmp.csv"):
    """ 
    train data should not contain columns of lable 
    both of the input data should not contain columns of id
    """
    if normal == True:
        train_data = data_clean_and_analysis.minmax(train_data)
        test_data = data_clean_and_analysis.minmax(test_data)
    gmm = GaussianMixture(n_components=2)
    gmm.fit(train_data)
    predicted = gmm.predict(test_data)
    return predicted

def i_svm(train_data, test_data,predicted,save = "output_record/tmp.csv"):
    train_x = train_data.iloc[:,:-1]
    train_y = train_data.iloc[:,-1]
    train_x = data_clean_and_analysis.minmax(train_x)
    test_data = data_clean_and_analysis.minmax(test_data)
    model = SVC(kernel='poly',C = 120)
    model.fit(train_x, train_y)
    predicted_svm = model.predict(test_data)
    count_svm = np.count_nonzero(predicted_svm == 1)>np.count_nonzero(predicted_svm == 0)
    count_gmm = np.count_nonzero(predicted == 1)>np.count_nonzero(predicted == 0)
    if count_gmm == count_svm:
        predicted[predicted == 1] = -1
        predicted[predicted == 0] = 1
        predicted[predicted == -1] = 0
    score, score_rate = data_clean_and_analysis.result_comparition(predicted)
    if score_rate > 0.5:
        predicted_df = pd.DataFrame(pd.read_csv('data/test.csv').iloc[:,-1])
        predicted_df['prediction'] = predicted
        predicted_df.to_csv(save, index=False)
    return score,score_rate