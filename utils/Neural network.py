import sklearn.metrics
import numpy as np
import pandas as pd
from sklearn import svm
import torch.nn as nn
import torch.optim as optim
import torch
from catboost import CatBoostClassifier, Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader


class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class TestData(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.input = nn.Linear(4, 16)
        self.hidden1 = nn.Linear(16, 8)
        self.hidden2 = nn.Linear(8, 4)
        self.output = nn.Linear(4, 1)

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.3)

        self.batch_norm_1 = nn.BatchNorm1d(32)
        self.batch_norm_2 = nn.BatchNorm1d(16)
        self.batch_norm_3 = nn.BatchNorm1d(8)
        self.batch_norm_4 = nn.BatchNorm1d(64)
        self.batch_norm_5 = nn.BatchNorm1d(4)
        self.batch_norm_6 = nn.BatchNorm1d(2)

        self.prelu = nn.PReLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, train_data):
        x = self.prelu(self.input(train_data))
        x = self.batch_norm_2(x)

        x = self.prelu(self.hidden1(x))
        x = self.batch_norm_3(x)
        x = self.dropout1(x)

        x = self.prelu(self.hidden2(x))
        x = self.batch_norm_4(x)
        x = self.dropout2(x)

        x = self.output(x)
        return x

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    accuracy = correct_results_sum / y_test.shape[0]
    accuracy = torch.round(accuracy * 100)

    return accuracy

if __name__ == '__main__':
    # * load the data
    train_data = pd.read_csv("./train.csv")
    train_data_origin = pd.read_csv("./train.csv")
    test_data = pd.read_csv("./test.csv")

    # train_data = train_data.query('feature_4 <= 23')
    X_train = train_data[['feature_1', 'feature_2', 'feature_3', 'feature_4']]
    y_train = train_data['label']

    # print(np.array(X_train))
    X_test = test_data[['feature_1', 'feature_2', 'feature_3', 'feature_4']]
    scale = MinMaxScaler()

    X_train = scale.fit_transform(X_train)
    X_test = scale.fit_transform(X_test)
    model = NeuralNetwork()
    model.to('cpu')

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    xTrain, xTest, yTrain, yTest = train_test_split(np.array(X_train), np.array(y_train), test_size=1/50)
    model.train()
    scaler = StandardScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.transform(xTest)
    Batch_Size = 32
    epoch_number = 100

    train_data_neural = TrainData(torch.FloatTensor(xTrain), torch.FloatTensor(yTrain))
    test_data_neural_1 = TestData(torch.FloatTensor(xTest))
    test_data_neural_output = TestData(torch.FloatTensor(np.array(X_test)))

    train_loader = DataLoader(dataset=train_data_neural, batch_size=Batch_Size, shuffle=True)
    test_loader_1 = DataLoader(dataset=test_data_neural_1, batch_size=1)
    test_loader_output = DataLoader(dataset=test_data_neural_output, batch_size=1)

    for epoch in range(1, epoch_number):
        total_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to('cpu'), y_batch.to('cpu')
            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            epoch_acc += acc.item()

        print(
            f'Epoch {epoch + 0:03}: | Loss: {total_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader_1:
            X_batch = X_batch.to('cpu')
            y_test_pred = model(X_batch)
            y_test_pred = y_test_pred.detach().numpy()
            y_test_pred[np.where(y_test_pred > 0)] = 1
            y_test_pred[np.where(y_test_pred < 0)] = 0
            y_pred_list.append(y_test_pred)

    # y_pred_list = [element.squeeze().tolist() for element in y_pred_list]

    y_output_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader_output:
            X_batch = X_batch.to('cpu')
            y_output_pred = model(X_batch)
            y_output_pred = y_output_pred.detach().numpy()
            y_output_pred[np.where(y_output_pred > 0)] = 1
            y_output_pred[np.where(y_output_pred < 0)] = 0
            y_output_list.append(y_output_pred)

    predict_dataset = pd.DataFrame({'example_id': test_data['example_id']}, columns=['example_id'])
    append_list = []
    accuracy_list = []

    for i in range(0, len(y_output_list)):
        append_list.append(int(y_output_list[i][0][0]))

    for i in range(0, len(y_pred_list)):
        accuracy_list.append((int(y_pred_list[i][0][0])))

    predict_dataset['prediction'] = append_list
    test_accuracy = np.array(accuracy_list)

    print(predict_dataset)
    df_concat = pd.concat([test_data, predict_dataset['prediction']], axis=1)
    print("before loc the df_concat is\n {}".format(df_concat))
    # df_concat.loc[df_concat.feature_4 >= 24, 'prediction'] = 1
    print("after loc the df_concat is\n {}".format(df_concat))
    predict_dataset = pd.DataFrame(df_concat, columns=['example_id', 'prediction'])
    # predict_dataset.loc[predict_dataset.prediction <= 0, 'prediction'] = int(1)
    # predict_dataset.loc[predict_dataset.prediction > 0, 'prediction'] = int(0)
    predict_dataset['prediction'].astype(int)
    predict_dataset = predict_dataset.convert_dtypes()
    predict_dataset.to_csv("./submission_neural.csv", index=False)
