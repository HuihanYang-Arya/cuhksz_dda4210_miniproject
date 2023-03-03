import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from utils import data_clean_and_analysis

def i_rf(train_data, test_data, save = "output_record/tmp.csv",train = True):
    """ 
    implementation of simple random forest algorithm
    """

    if train == True:
        train_x,train_y,val_x,val_y,test_data = data_clean_and_analysis.split_normal(train_data,test_data)
    else:
        train_x = train_data.iloc[:,:-1]
        train_y = train_data.iloc[:,-1]
        train_x = data_clean_and_analysis.minmax(train_x)
        test_data = data_clean_and_analysis.minmax(test_data)
    #train the model
    clf = RandomForestClassifier(n_estimators = 500)
    clf.fit(train_x,train_y.values)
    if train == True:
        clf = clf.fit(train_x, train_y)
        return clf.score(val_x,val_y)
    else:
        clf = clf.fit(train_x, train_y)
        predict = clf.predict(test_data)
        predicted_df = pd.DataFrame(pd.read_csv('data/augmented_test.csv').iloc[:,-1])
        predicted_df['prediction'] = predict
        predicted_df.to_csv(save, index=False)

def small_rf(train_x,train_y,val_x,val_y,test_data, column_list,n_estimator=100):
    clf = RandomForestClassifier(n_estimators=n_estimator,max_features='auto')
    clf.fit(train_x.iloc[:,column_list],train_y.values)
    score_val = clf.score(val_x.iloc[:,column_list],val_y)
    predict = clf.predict(test_data.iloc[:,column_list])
    return score_val,predict


def rf_ensamble(train_data, test_data, save = "output_record/tmp.csv",train = True):
    """ 
    implementation of ensamble random forest algorithm
    """
    if train == True:
        train_x,train_y,val_x,val_y,test_data = data_clean_and_analysis.split_normal(train_data,test_data)
    else:
        train_x = train_data.iloc[:,:-1]
        train_y = train_data.iloc[:,-1]
        train_x = data_clean_and_analysis.minmax(train_x)
        test_data = data_clean_and_analysis.minmax(test_data)
    clf9 = RandomForestClassifier()
    clf2 = RandomForestClassifier(max_features=1)
    clf3 = RandomForestClassifier(max_features=2)
    clf4 = RandomForestClassifier(max_features=3)
    clf1 = RandomForestClassifier(max_features=4)
    clf5 = RandomForestClassifier(criterion='entropy')
    clf6 = RandomForestClassifier(max_features=1)
    clf7 = RandomForestClassifier()
    clf8 = RandomForestClassifier()
    clf = VotingClassifier(estimators=[('svr',clf1), ('rf', clf2), ('svcp', clf3),('svcr',clf4),('gn',clf5),('svcl',clf6),('svcs',clf7),('111',clf8),('1',clf9)], voting='soft')
    if train == True:
        clf = clf.fit(train_x, train_y)
        return clf.score(val_x,val_y)
    else:
        clf = clf.fit(train_x, train_y)
        predict = clf.predict(test_data)
        predicted_df = pd.DataFrame(pd.read_csv('data/augmented_test.csv').iloc[:,-1])
        predicted_df['prediction'] = predict
        predicted_df.to_csv(save, index=False)
    
    
