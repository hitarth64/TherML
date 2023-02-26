import torch
import numpy as np
import utilities as utils
from torch.utils.data import Dataset, DataLoader, TensorDataset

class TrainData(Dataset):

    def __init__(self, df, max_round, min_round, featurization, model, transform=None, train_mode=False, feature_reduction=None):
        """
        df: dataframe of featurized data
        max_round: maximum round to be included
        min_round: minimum round to be included
        featurization: One of the three featurization functions available in ```utilities```
        model: Trained prior model from one of the 3 possibilities: ['roost', 'crabnet', 'magpie']
        transform: transformation function
        train_mode: True only if transform is not None
                    This trains the ```transforms``` required for pre-processing
        feature_reduction: Defaults to None; can be a dimension reduction component
        """
        self.max_round = max_round
        self.min_round = min_round
        self.transform = transform
        self.df = df[df['max_round'] <= self.max_round]
        self.df = self.df[self.df['max_round'] >= self.min_round]

        compounds1_set = set(self.df['c1'].values)
        compounds2_set = set(self.df['c2'].values)
        compounds_list = list(compounds1_set.union(compounds2_set))

        X , _ = featurization(compounds_list)
        feats = dict(zip(compounds_list, X))

        X1, X2, state, y = [], [], [], []

        for i in self.df.index:
            X1.append(feats[self.df.loc[i,'c1']].astype('float32').tolist())
            X2.append(feats[self.df.loc[i,'c2']].astype('float32').tolist())

        X1, X2 = np.array(X1).astype('float32'), np.array(X2).astype('float32')
        if feature_reduction is not None:
            self.df['predicted_pf1'] = model.predict(feature_reduction.transform(np.concatenate([X1,self.df['T1'].values.reshape(-1,1)],axis=1)))
            self.df['predicted_pf2'] = model.predict(feature_reduction.transform(np.concatenate([X2,self.df['T2'].values.reshape(-1,1)],axis=1)))
        else:
            self.df['predicted_pf1'] = model.predict(np.concatenate([X1,self.df['T1'].values.reshape(-1,1)],axis=1))
            self.df['predicted_pf2'] = model.predict(np.concatenate([X2,self.df['T2'].values.reshape(-1,1)],axis=1))

        state = self.df[['T1', 'T2', 'predicted_pf1', 'predicted_pf2']].values.astype('float32')
        y = self.df['pf1>pf2'].values.astype('float32')

        if train_mode:
            assert transform is not None, "Transform must be provided if train-mode is TRUE"
            X1 = self.transform[0].fit_transform(X1)
            X2 = self.transform[1].fit_transform(X2)
            state = self.transform[2].fit_transform(state)

        elif train_mode==False and transform is not None:
            X1 = self.transform[0].transform(X1)
            X2 = self.transform[1].transform(X2)
            state = self.transform[2].transform(state)

        self.dataset = TensorDataset(torch.Tensor(X1), torch.Tensor(X2), torch.Tensor(state), torch.tensor(y))

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__ (self):
        return len(self.dataset)


class Inquire(Dataset):

    def __init__(self, Candidates, Temperatures, BestFromLastRound, ReferenceTemperature, Transforms, featurization, PriorModel, feature_reduction=None):
        """
        Candidates: List of pymatgen compositions: candidates to explore
        Temperatures: List of temperatures in Kelvin
        BestFromLastRound: Pymatgen composition of the best candidate from the last round
        ReferenceTemperature: Temperature used for the ```BestFromLastRound``` material for comparison
        Transforms: transforms for preprocessing the inputs. These must be trained on training data
        featurization: corresponding featurization function from utilities
        PriorModel: Prior model being used
        feature_reduction: Defaults to none; used to perform dimensionality reduction; a scikit-learn object with .transform routine
        """
        self.candidates = Candidates
        self.temperatures = Temperatures
        self.lastbest = BestFromLastRound
        self.transform = Transforms
        self.reference_temperature = ReferenceTemperature
        self.features, self.formation_energies = featurization(self.candidates + [self.lastbest])

        if feature_reduction is not None:
            self.features = np.concatenate([self.features, np.array(self.temperatures + [self.reference_temperature]).reshape(-1,1)], axis=1)
            self.reduced_features = feature_reduction.transform(self.features)
            self.predicted_pf = PriorModel.predict(self.reduced_features).reshape(-1,)
        else:
            self.features = np.concatenate([self.features, np.array(self.temperatures + [self.reference_temperature]).reshape(-1,1)], axis=1)
            self.predicted_pf = PriorModel.predict(self.features).reshape(-1,)

        self.x = self.features[:,:-1]
        self.T = self.features[:,-1].reshape(-1,)
        self.formula = [c.formula for c in Candidates] + [BestFromLastRound.formula]
        #self.formation_energies = list(self.formation_energies[1])[0][:,0].cpu().detach().numpy()
        #self.formation_energies = dict(zip(self.formula, self.formation_energies))
        self.groups = list(np.arange(len(Candidates)+1).astype('int32'))

        X1, X2, state, groups = [], [], [], []
        for i in range(len(self.candidates)+1):
            for j in range(i+1, len(self.candidates)+1):
                X1.append(list(self.x[i]))
                X2.append(list(self.x[j]))
                state.append([self.T[i], self.T[j], self.predicted_pf[i], self.predicted_pf[j]])
                groups.append([self.groups[i], self.groups[j]])

        X1 = self.transform[0].transform(X1)
        X2 = self.transform[1].transform(X2)
        state = self.transform[2].transform(state)
        groups = np.array(groups)

        self.dataset = TensorDataset(torch.Tensor(X1), torch.Tensor(X2), torch.Tensor(state), torch.Tensor(groups))

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__ (self):
        return len(self.dataset)

