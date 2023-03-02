import pandas as pd
from sklearn.svm import SVC
from utils import data_clean_and_analysis
from sklearn.ensemble import  AdaBoostClassifier


def svm_ensamble(train_data, test_data, save = "output_record/tmp.csv",n_estimator = 6,learning_rate = 0.2,train = True
                 ):
    if train == True:
    #random split
        train_x,train_y,val_x,val_y,test_data = data_clean_and_analysis.split_normal(train_data,test_data)
    else:
        train_x = train_data.iloc[:,:-1]
        train_y = train_data.iloc[:,-1]
        train_x = data_clean_and_analysis.minmax(train_x)
        test_data = data_clean_and_analysis.minmax(test_data)
    baseSVM = SVC(probability=True, kernel='rbf', class_weight='balanced')
    model = AdaBoostClassifier(n_estimators=n_estimator, random_state=42, learning_rate=learning_rate, base_estimator=baseSVM)
    if train == True:
        model.fit(train_x, train_y)
        return model.score(val_x,val_y)
    else:
        model.fit(train_x, train_y)
        predict = model.predict(test_data)
        score, score_rate = data_clean_and_analysis.result_comparition(predict)
        predicted_df = pd.DataFrame(pd.read_csv('data/test.csv').iloc[:,-1])
        predicted_df['prediction'] = predict
        predicted_df.to_csv(save, index=False)
        return score,score_rate
    



    
    