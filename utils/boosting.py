import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from utils import data_clean_and_analysis

def i_boosting(train_data, test_data, train = True, save = "output_record/tmp.csv"):
    """ 
    both of the input data should not contain columns of id
    """
    if train == True:
        train_x,train_y,val_x,val_y,test_data = data_clean_and_analysis.split_normal(train_data,test_data)
    else:
        train_x = train_data.iloc[:,:-1]
        train_y = train_data.iloc[:,-1]
        train_x = data_clean_and_analysis.minmax(train_x)
        test_data = data_clean_and_analysis.minmax(test_data)
    #train the model
    eclf1 = AdaBoostClassifier(n_estimators = 400)
    if train == True:
        eclf1 = eclf1.fit(train_x, train_y)
        return eclf1.score(val_x,val_y)
    else:
        eclf1 = eclf1.fit(train_x, train_y)
        predict = eclf1.predict(test_data)
        predicted_df = pd.DataFrame(pd.read_csv('data/augmented_test.csv').iloc[:,-1])
        predicted_df['prediction'] = predict
        predicted_df.to_csv(save, index=False)