import pandas as pd
from sklearn.svm import SVC
from utils import data_clean_and_analysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier,AdaBoostClassifier

def i_gb(train_data, test_data, save = "output_record/tmp.csv",train = True):
    """
    Implementation of gaussian bayes.
    """
    if train == True:
        train_x,train_y,val_x,val_y,test_data = data_clean_and_analysis.split_normal(train_data,test_data)
    else:
        train_x = train_data.iloc[:,:-1]
        train_y = train_data.iloc[:,-1]
        train_x = data_clean_and_analysis.minmax(train_x)
        test_data = data_clean_and_analysis.minmax(test_data)
    clf = GaussianNB()
    if train == True:
        clf = clf.fit(train_x, train_y)
        return clf.score(val_x,val_y)
    else:
        clf = clf.fit(train_x, train_y)
        predict = clf.predict(test_data)
        predicted_df = pd.DataFrame(pd.read_csv('data/augmented_test.csv').iloc[:,-1])
        predicted_df['prediction'] = predict
        predicted_df.to_csv(save, index=False)