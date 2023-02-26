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

# HYPERPARAMETERS
learning_rate = 1e-3
epochs = 100
batch_size = 32
n_splits = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "saved_models/updater.pth"

featurization = utils.featurize_magpie
priormodel = pd.read_pickle('prior_models/model_magpie_tpot')[0]

# DATA
df = pd.read_pickle('featurized_experimental_data.pkl')
df['max_round'] = df[['round1','round2']].max(axis=1).values
df['T1'] = df['T1'] + 273 # convert to kelvin
df['T2'] = df['T2'] + 273 # convert to kelvin

kfold = KFold(n_splits=n_splits, shuffle=True)
train_data = data.TrainData(df, 3, 1, featurization=featurization, model=priormodel, transform=[StandardScaler(), StandardScaler(), StandardScaler()], train_mode=True)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

for batch in train_loader:
    break

#inquiry_data = data.Inquire([Composition('Pb0.98Se0.98Sn0.02Sb0.02'), Composition('Pb0.99Se0.99Sn0.01Sb0.01'), Composition('Pb0.95Se0.95Sn0.05Sb0.05'), Composition('Pb0.995Se0.995Sn0.005Sb0.005'), Composition('PbSe'), Composition('SnSb')], [323,323,323,323,323,323], Composition('SnSe'), 323, train_data.transform, featurization, priormodel)

inquiry_data = data.Inquire([Composition('Pb0.99Se0.99Sn0.01Sb0.01')]*10, list(np.linspace(300,550,10)), Composition('PbSe'), 300, train_data.transform, featurization, priormodel)

inquiry_loader = DataLoader(dataset=inquiry_data, batch_size=batch_size)

#model = models.ModelSiamese(batch[0].shape[1], batch[2].shape[1], units=3849, layers=1)
model = models.Modelv1(batch[0].shape[1] + batch[1].shape[1] + batch[2].shape[1], units=1969, layers=4) 
model.to(device)

model.load_state_dict(torch.load(MODEL_PATH))
predictions = utils.MakePredictions(model, inquiry_loader, device)
print(predictions)
