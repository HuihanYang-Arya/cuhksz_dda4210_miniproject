import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from utils import data_clean_and_analysis
from sklearn.model_selection import train_test_split

def i_boosting(train_data, test_data, normal = True, save = "output_record/tmp.csv"):
    """ 
    both of the input data should not contain columns of id
    """
    #random split to see the score
    train_tr_data,val_data = train_test_split(train_data,test_size=0.15,train_size=0.85)

    train_y = train_tr_data.iloc[:,-1]
    train_x = train_tr_data.iloc[:,:-1]
    val_y = val_data.iloc[:,-1]
    val_x = val_data.iloc[:,:-1]
    if normal == True:
        train_x = data_clean_and_analysis.minmax(train_x)
        val_x = data_clean_and_analysis.minmax(val_x)
        test_data = data_clean_and_analysis.minmax(test_data)
    #train the model
    clf = AdaBoostClassifier()
    clf.fit(train_x,train_y.values)
    print(clf.base_estimator_)
    print(clf.estimator_weights_)
    print(clf.feature_importances_)
    score_val = clf.score(val_x,val_y)
    predicted = clf.predict(test_data)
    score, score_rate = data_clean_and_analysis.result_comparition(predicted)
    predicted_df = pd.DataFrame(pd.read_csv('data/test.csv').iloc[:,-1])
    predicted_df['prediction'] = predicted
    predicted_df.to_csv(save, index=False)
    return score,score_rate,score_val