import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from utils import data_clean_and_analysis

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
    score, score_rate = data_clean_and_analysis.result_comparition(predicted)
    if score_rate < 0.1:
        predicted_df = pd.DataFrame(pd.read_csv('data/test.csv').iloc[:,-1])
        predicted_df['prediction'] = predicted
        predicted_df.to_csv(save, index=False)
    return score,score_rate

def gmm_plot(predicted_data:pd.DataFrame,test_data:pd.DataFrame):
    colors = ["navy", "turquoise"]
    tmp = predicted_data.iloc[:,1]
    test_data['useless_feature'] = np.zeros(len(tmp))
    test_data['predicted'] = tmp
    test_matrix = test_data.values
    for i, color in enumerate(colors):
        data = test_matrix[test_matrix[:,2] == i][:,0:2]
        plt.scatter(data[:, 0], data[:, 1], color=color)
        plt.show()


