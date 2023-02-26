# Training script
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import roost_feature_generation
import sys
sys.path.append('/home/mxyptlkr/scratch/CrabNet')
import crabnet_feature_extraction
from matminer.featurizers.composition import ElementProperty

def train_model(model, dataloader, optimizer, criterion, device, epochno, ReturnValues=False):
    epoch_loss = 0.
    epoch_acc = 0.
    epoch_f1 = 0.
    no_of_batches = 0
    Global_cm = np.zeros((2,2))

    model.train()
    for x1_batch, x2_batch, x3_batch, y_batch in dataloader:

        x1_batch, x2_batch, x3_batch, y_batch = x1_batch.to(device), x2_batch.to(device), x3_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(x1_batch, x2_batch, x3_batch)

        loss = criterion(pred, y_batch.unsqueeze(1))
        acc, f1, cm = accuracy_function(pred, y_batch.unsqueeze(1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc
        epoch_f1 += f1
        Global_cm += cm
        no_of_batches += 1

    if epochno % 1 == 0:
        print('Epoch no: {}, | Loss: {:.4f} | Accuracy: {:.4f} | F1-score: {:.4f} | CM: {:.0f} {:.0f} {:.0f} {:.0f}'\
            .format(epochno, epoch_loss/len(dataloader), epoch_acc/no_of_batches, epoch_f1/no_of_batches, \
                    Global_cm[0,0], Global_cm[0,1], Global_cm[1,0], Global_cm[1,1]))

    if ReturnValues:
        return epoch_f1/no_of_batches


def accuracy_function(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred)).detach().cpu().numpy()
    y_test_cpu = y_test.detach().cpu().numpy()
    f1, acc = f1_score(y_test_cpu, y_pred_tag), accuracy_score(y_test_cpu, y_pred_tag)
    cm = confusion_matrix(y_test_cpu, y_pred_tag)
    return acc, f1, cm

def evaluate_testdata(model, dataloader, device):
    pred, true = [], []
    model.eval()
    with torch.no_grad():
        for x1_batch, x2_batch, x3_batch, y_batch in dataloader:
            x1_batch, x2_batch, x3_batch, y_batch = x1_batch.to(device), x2_batch.to(device), x3_batch.to(device), y_batch.to(device)
            y_pred = model(x1_batch, x2_batch, x3_batch)
            y_pred = torch.sigmoid(y_pred)
            y_pred = torch.round(y_pred).detach().cpu().numpy()
            pred += list(y_pred[:,0])
            true += list(y_batch.detach().cpu().numpy())

    acc, f1 = accuracy_score(pred, true), f1_score(pred, true)
    print('Accuracy: {:.4f} | F1-score: {:.4f}'.format(acc, f1))
    return acc, f1


def featurize_roost(list_of_compositions):
    dff = pd.DataFrame(columns=['composition','PFn_max'])
    dff['composition'] = [i.fractional_composition.formula for i in list_of_compositions]
    dff['PFn_max'] = -1
    dff.index.name = 'material_id'
    dff.to_csv('tmp_data.csv')
    feats, formation_energies = roost_feature_generation.return_function('tmp_data.csv')
    return feats, formation_energies

def featurize_crab(list_of_compositions):
    crabnet_df = pd.DataFrame(columns=['formula','target'])
    crabnet_df['formula'] = [i.fractional_composition.formula for i in list_of_compositions]
    crabnet_df['target'] = 0.0
    crabnet_df.to_csv('/home/mxyptlkr/scratch/CrabNet/data/power_factor/val.csv')
    feats = crabnet_feature_extraction.get_features()
    return feats, None # second argument is just to make things consistent

def featurize_magpie(list_of_compositions):
    magpie_df = pd.DataFrame(columns=['comps','target'])
    magpie_df['comps'] = list_of_compositions
    magpie_df['target'] = 0.0
    ep = ElementProperty.from_preset('magpie')
    feats = ep.fit_featurize_dataframe(magpie_df, col_id='comps').drop(columns=magpie_df.columns.values).values
    return feats, None # second argument is just to make things consistent

def MakePredictions(model, dataloader, device):
    formula = dataloader.dataset.formula
    Ts = dataloader.dataset.T
    #formation_energies = dataloader.dataset.formation_energies
    pred, label = [], []
    model.eval()
    with torch.no_grad():
        for x1,x2,state,group in dataloader:
            x1, x2, state = x1.to(device), x2.to(device), state.to(device)
            y_pred = model(x1, x2, state)
            y_labels = torch.round(torch.sigmoid(y_pred)).detach().cpu().numpy()
            pred += list(y_labels[:,0])
            label += [(formula[int(i)],formula[int(j)],Ts[int(i)],Ts[int(j)]) for i,j in group.cpu().numpy()]
    comparisons = dict(zip(label, pred))
    rankings = {}

    for k in comparisons.keys():
        if comparisons[k] == 1:
            if (k[0],k[2]) in rankings.keys():
                rankings[(k[0],k[2])] += 1
            else:
                rankings[(k[0],k[2])] = 1
        elif comparisons[k] == 0:
            if (k[1],k[3]) in rankings.keys():
                rankings[(k[1],k[3])] += 1
            else:
                rankings[(k[1],k[3])] = 1

    output = pd.DataFrame(columns=['formula','temperature','values'])
    key_value_pairs = rankings.items()
    formula_list = [k[0][0] for k in key_value_pairs]
    temperature_list = [k[0][1] for k in key_value_pairs]
    values_list = [k[1] for k in key_value_pairs]
    output['formula'] = formula_list
    output['values'] = values_list
    output['temperature'] = temperature_list
    output = output.sort_values(by=['values'])
    return output
