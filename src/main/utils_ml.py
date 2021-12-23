import os, random, glob, csv, pickle 
import utils

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler

def analyse_ml(path, firstAudioName):
    """
    This function is called to compute easily all the features of signals.
    Because of the cepstrum and autocorrelation pitch estimation requirements, path must point to
    a directory where minimum 5 audiofiles of a speaker are stored.
    Adapted to csv files creation
    """
    os.chdir(path)
    globfiles = random.sample(glob.glob("*.wav"), 4)
    files = glob.glob(firstAudioName) + globfiles
    print(files)
    autocorr_pitch = utils.autocorrelation_pitch_estim(files)
    cepstrum_pitch = utils.cepstrum_pitch_estim(files)
    formants_list = []
    for file in files:
        formants = utils.compute_formants(file)
        for f in formants:
            formants_list.append(f)
    
    f1_list = []
    f2_list = []
    for i in range(len(formants_list)):
        if (formants_list[i][0] > 90 and formants_list[i][0] < 1000):
            f1_list.append(formants_list[i][0])
        if (formants_list[i][1] > 600 and formants_list[i][1] < 3200):
            f2_list.append(formants_list[i][1])
    os.chdir("../../")
    return autocorr_pitch, cepstrum_pitch, f1_list, f2_list


def create_TrainingCSV():
    """
    This function is called to easily create a csv with training data.
    This csv will then be passed to the AI to train its model.
    """
    header = ['Autocorrelation_Pitch', 'Cepstrum_Pitch', 'F1', 'F2', 'Sexe']
    data = []

    print("START")

    for filename in os.listdir("data/bdl_a/"): #Homme
        autocorr_pitch, cepstrum_pitch, f1_list, f2_list = analyse_ml("data/bdl_a", filename)
        res = [autocorr_pitch, cepstrum_pitch, np.mean(f1_list), np.mean(f2_list), 1]
        data.append(res)
    
    for filename in os.listdir("data/bdl_b/"): #Homme
        autocorr_pitch, cepstrum_pitch, f1_list, f2_list = analyse_ml("data/bdl_b", filename)
        res = [autocorr_pitch, cepstrum_pitch, np.mean(f1_list), np.mean(f2_list), 1]
        data.append(res)

    for filename in os.listdir("data/slt_a/"): #Femme
        autocorr_pitch, cepstrum_pitch, f1_list, f2_list = analyse_ml("data/slt_a", filename)
        res = [autocorr_pitch, cepstrum_pitch, np.mean(f1_list), np.mean(f2_list), 0]
        data.append(res)
    
    for filename in os.listdir("data/slt_b/"): #Femme
        autocorr_pitch, cepstrum_pitch, f1_list, f2_list = analyse_ml("data/slt_b", filename)
        res = [autocorr_pitch, cepstrum_pitch, np.mean(f1_list), np.mean(f2_list), 0]
        data.append(res)

    with open('training.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    
    print("FINISH")

def create_TestCSV():
    """
    This function is called to easily create a csv with testing data.
    This csv will then be passed to the AI to test its model.
    """
    header = ['Autocorrelation_Pitch', 'Cepstrum_Pitch', 'F1', 'F2', 'Sexe']
    data = []

    print("START")

    for filename in os.listdir("data/rms_a/"): #Homme
        autocorr_pitch, cepstrum_pitch, f1_list, f2_list = analyse_ml("data/rms_a", filename)
        res = [autocorr_pitch, cepstrum_pitch, np.mean(f1_list), np.mean(f2_list), 1]
        data.append(res)
    
    for filename in os.listdir("data/rms_b/"): #Homme
        autocorr_pitch, cepstrum_pitch, f1_list, f2_list = analyse_ml("data/rms_b", filename)
        res = [autocorr_pitch, cepstrum_pitch, np.mean(f1_list), np.mean(f2_list), 1]
        data.append(res)

    for filename in os.listdir("data/cms_a/"): #Femme
        autocorr_pitch, cepstrum_pitch, f1_list, f2_list = analyse_ml("data/cms_a", filename)
        res = [autocorr_pitch, cepstrum_pitch, np.mean(f1_list), np.mean(f2_list), 0]
        data.append(res)
    
    for filename in os.listdir("data/cms_b/"): #Femme
        autocorr_pitch, cepstrum_pitch, f1_list, f2_list = analyse_ml("data/cms_b", filename)
        res = [autocorr_pitch, cepstrum_pitch, np.mean(f1_list), np.mean(f2_list), 0]
        data.append(res)

    with open('test.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    
    print("FINISH")

###Machile learning algorithm : Binary Classification###

##Train data
class TrainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


##Test data    
class TestData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)

##Classification
class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Number of input features is 4.
        self.layer_1 = nn.Linear(4, 64) 
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def train_BinaryClassificationModel():
    #Read CSV
    training_data = pd.read_csv("data/ml_data/training.csv") #Sexe : Men = 1 | Women = 0
    test_data = pd.read_csv("data/ml_data/test.csv") #Sexe : Men = 1 | Women = 0

    #We choose input features and label in csv files for the training and the test 
    X_train = training_data.iloc[:, :-1]
    y_train = training_data.iloc[:, -1]
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    scaler = StandardScaler() #Normalizes inputs
    X_train = scaler.fit_transform(X_train.values.tolist())
    X_test = scaler.transform(X_test)

    #Hyperparameters
    EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001

    train_data = TrainData(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_data = TestData(torch.FloatTensor(X_test))

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = BinaryClassification()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss() #Cost function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()

    for e in range(1, EPOCHS+1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            y_pred = model(X_batch) #Y'
            
            loss = criterion(y_pred, y_batch.unsqueeze(1)) #Error
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))
            
            loss.backward()
            optimizer.step() #Modification of the bias and weights
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            

        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    return model, scaler

def save_BinaryClassificationModel(model, scaler):
    torch.save(model.state_dict(), "data/ml_data/BinaryClassificationModel.pt")
    fileScaler = open("data/ml_data/BinaryClassificationScaler.pt", 'wb') 
    pickle.dump(scaler, fileScaler)


def load_BinaryClassificationModel(pathModel, pathScaler):
    model = BinaryClassification()
    model.load_state_dict(torch.load(pathModel))
    model.eval()
    fileScaler = open(pathScaler, 'rb')
    scaler = pickle.load(fileScaler)

    return model, scaler

def useModel(model, scaler, inputs):
    inputs = [inputs]
    x = scaler.transform(inputs)

    x = torch.FloatTensor(x) #Convert to tensor for compatibility with the model
    model.eval() 

    with torch.no_grad(): #Obtaining the prediction
        y = model(x)
        y = torch.round(torch.sigmoid(y).squeeze(0)).item()
        if y: 
            print('It\'s a Man')
        else: 
            print('It\'s a Woman')

if __name__ == "__main__":
    #Create CSV
    #create_TestCSV()
    #create_TrainingCSV()

    model, scaler = train_BinaryClassificationModel()
    save_BinaryClassificationModel(model, scaler)
    model, scaler = load_BinaryClassificationModel("data/ml_data/BinaryClassificationModel.pt", "data/ml_data/BinaryClassificationScaler.pt")
    useModel(model, scaler, [110.074607, 108.019277, 322.401576, 1873.313785]) #H
    useModel(model, scaler, [188.91156, 243.643789, 283.640021, 1937.04209]) #F

    