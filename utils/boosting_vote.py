import pandas as pd
from sklearn.svm import SVC
from utils import data_clean_and_analysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier,AdaBoostClassifier

def i_vb(train_data, test_data, save = "output_record/tmp.csv",train = True):
    """
    implementation of vote boosting algorithm
    """
    if train == True:
        train_x,train_y,val_x,val_y,test_data = data_clean_and_analysis.split_normal(train_data,test_data)
    else:
        train_x = train_data.iloc[:,:-1]
        train_y = train_data.iloc[:,-1]
        train_x = data_clean_and_analysis.minmax(train_x)
        test_data = data_clean_and_analysis.minmax(test_data)
    clf1 = RandomForestClassifier(max_features=2)
    clf2 = GaussianNB()
    clf3 = RandomForestClassifier()
    clf4 = RandomForestClassifier()
    clf5 = AdaBoostClassifier()
    clf6 = RandomForestClassifier(max_features=3)
    clf7 = SVC(probability = True,kernel='rbf',C = 8)
    clf8 = SVC(probability = True,kernel='rbf',C = 8)
    clf9 = RandomForestClassifier()
    eclf1 = VotingClassifier(estimators=[('svr',clf1), ('rf', clf2), ('svcp', clf3),('svcr',clf4),('gn',clf5),('svcl',clf6),('1',clf7),('13',clf8),('14',clf9)], voting='hard')
    if train == True:
        eclf1 = eclf1.fit(train_x, train_y)
        return eclf1.score(val_x,val_y)
    else:
        eclf1 = eclf1.fit(train_x, train_y)
        predict = eclf1.predict(test_data)
        predicted_df = pd.DataFrame(pd.read_csv('data/augmented_test.csv').iloc[:,-1])
        predicted_df['prediction'] = predict
        predicted_df.to_csv(save, index=False)

    

