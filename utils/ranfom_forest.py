import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from utils import data_clean_and_analysis
from sklearn.model_selection import train_test_split

def i_rf(train_data, test_data, normal = True, save = "output_record/tmp.csv"):
    """ 
    both of the input data should not contain columns of id
    """
    #random split to see the score
    train_x,train_y,val_x,val_y,test_data = data_clean_and_analysis.split_normal(train_data,test_data)
    #train the model
    clf = RandomForestClassifier(n_estimators = 500)
    clf.fit(train_x,train_y.values)
    score_val = clf.score(val_x,val_y)
    predicted = clf.predict(test_data)
    score, score_rate = data_clean_and_analysis.result_comparition(predicted)
    predicted_df = pd.DataFrame(pd.read_csv('data/test.csv').iloc[:,-1])
    predicted_df['prediction'] = predicted
    predicted_df.to_csv(save, index=False)
    return score,score_rate,score_val

def small_rf(train_x,train_y,val_x,val_y,test_data, column_list,n_estimator=100):
    clf = RandomForestClassifier(n_estimators=n_estimator,max_features='auto')
    clf.fit(train_x.iloc[:,column_list],train_y.values)
    score_val = clf.score(val_x.iloc[:,column_list],val_y)
    predict = clf.predict(test_data.iloc[:,column_list])
    return score_val,predict


def rf_ensamble(train_data, test_data, normal=True, save = "output_record/tmp.csv"):
    #random split
    train_x,train_y,val_x,val_y,test_data = data_clean_and_analysis.split_normal(train_data,test_data)
    predict = np.zeros((7,len(test_data)))
    socre1,predict[0,:] = small_rf(train_x,train_y,val_x,val_y,test_data,[2,3])
    socre2,predict[1,:] = small_rf(train_x,train_y,val_x,val_y,test_data,[1,2])
    socre3,predict[2,:] = small_rf(train_x,train_y,val_x,val_y,test_data,[1,3])
    socre4,predict[3,:] = small_rf(train_x,train_y,val_x,val_y,test_data,[0,1])
    socre5,predict[4,:] = small_rf(train_x,train_y,val_x,val_y,test_data,[0,2])
    socre6,predict[5,:] = small_rf(train_x,train_y,val_x,val_y,test_data,[0,3])
    socre7,predict[6,:] = small_rf(train_x,train_y,val_x,val_y,test_data,[0,1,2,3],n_estimator=200)
    predict[6:] = predict[6,:]*1.5
    predict_sum = np.sum(predict,axis = 0)
    predict = np.where(predict_sum>3.5,1,0)
    score, score_rate = data_clean_and_analysis.result_comparition(predict)
    return score,score_rate
    
    
