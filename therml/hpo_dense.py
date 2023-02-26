import comet_ml
from comet_ml import Optimizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import models
import utilities as utils
import data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset, SubsetRandomSampler
from pymatgen.core.composition import Composition
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, pearsonr

# RANGE OF HYPERPARAMETERS
config = {
    "algorithm": "bayes",

    "parameters": {
        "units": {"type": "integer", "min": 10, "max": 4000},
        "layers": {"type": "integer", "min": 1, "max": 20},
        "learning_rate": {"type": "float", "min": 1e-5, "max": 1e-2, "scalingType":"loguniform"},
        "batch_size": {"type": "integer", "min": 16, "max": 256},
    },

    "spec": {
        "metric": "combined_metric",
        "objective": "minimize",
        "seed": 1234
    },
}

# Provide your api-key
opt = Optimizer(config, project_name="therml", api_key="api-key",workspace="your-username")

# HYPERPARAMETERS
learning_rate = 1e-3
epochs = 100
batch_size = 32
n_splits = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'updater.pth'
featurization = utils.featurize_magpie
priormodel = pd.read_pickle('prior_models/model_magpie_tpot')[0]

# DATA
df = pd.read_pickle('featurized_experimental_data.pkl')
df['max_round'] = df[['round1','round2']].max(axis=1).values
df['T1'] = df['T1'] + 273 # convert to kelvin
df['T2'] = df['T2'] + 273 # convert to kelvin

kfold = KFold(n_splits=n_splits, shuffle=True)
train_data = data.TrainData(df, 3, 1, featurization=featurization, model=priormodel, transform=[StandardScaler(), StandardScaler(), StandardScaler()], train_mode=True)
val_data = data.TrainData(df, 3, 1, featurization=featurization, model=priormodel, transform=train_data.transform, train_mode=False)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

# looping through just once to get the input sizes
for batch in train_loader:
    break

def evaluate_hparams(units, layers, learning_rate, batch_size):

    model = models.Modelv1(batch[0].shape[1] + batch[1].shape[1] + batch[2].shape[1], units=units, layers=layers)
    #model = models.ModelSiamese(batch[0].shape[1], batch[2].shape[1], units=units, layers=layers)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    cv_accuracy, cv_f1score = [], []
    for fold,(train_idx,test_idx) in enumerate(kfold.split(train_data)):
        print('====== Fold no: {} ========'.format(fold))
        train_subsampler = SubsetRandomSampler(train_idx)
        test_subsampler = SubsetRandomSampler(test_idx)
        trainloader = DataLoader(train_data, batch_size=batch_size, sampler=train_subsampler)
        testloader = DataLoader(val_data, batch_size=batch_size, sampler=test_subsampler) # to prevent transforms being fit on the unseen fold
        
        model.apply(models.reset_weight)

        for epochno in range(epochs):
            utils.train_model(model, trainloader, optimizer, criterion, device, epochno)

        acc, f1 = utils.evaluate_testdata(model, testloader, device)
        cv_accuracy.append(acc)
        cv_f1score.append(f1)

    print(f"{n_splits} CV accuracy: {np.mean(cv_accuracy):.4f} | f1-score: {np.mean(cv_f1score):.4f}")
    
    # Once cross-validation score has been evaluated, train on all the folds. This model will be used for making inferences
    model.apply(models.reset_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epochno in range(epochs):
        trainingscore = utils.train_model(model, train_loader, optimizer, criterion, device, epochno, ReturnValues=True)

    MODEL_PATH = f"unit_{units:.0f}_layers_{layers:.0f}_lr_{learning_rate:.5f}_batch_size_{batch_size:.0f}_model.pth"
    torch.save(model.state_dict(), 'saved_models/' + MODEL_PATH)

    return np.mean(cv_accuracy), np.mean(cv_f1score), trainingscore


for experiment in opt.get_experiments():

    experiment.add_tag('hpo_dense_magpie')

    accuracy, f1score, training_f1 = evaluate_hparams(experiment.get_parameter('units'), \
                                                             experiment.get_parameter('layers'), \
                                                             experiment.get_parameter('learning_rate'),\
                                                             experiment.get_parameter('batch_size'))
    experiment.log_metric('cv_accuracy', accuracy)
    experiment.log_metric('cv_f1score', f1score)
    experiment.log_metric('Training_f1_score', final_f1)
