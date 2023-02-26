import os
import numpy as np
from sklearn.ensemble import *
from sklearn.model_selection import *

from datetime import datetime
import sys
sys.path.append('../roost/')

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        r2_score,
        roc_auc_score,
)

from roost.roost.model import Roost
from roost.roost.data import CompositionData, collate_batch
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss, NLLLoss
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from roost.core import Normalizer, RobustL1Loss, RobustL2Loss, sampled_softmax
from roost.segments import ResidualNetwork

data_params = {
                "batch_size": 16,
                "num_workers": 0,
                "pin_memory": False,
                "shuffle": False,
                "collate_fn": collate_batch,}

device=torch.device("cpu")
transfer= '../roost/models/oqmd_100_epochs_model/checkpoint-r0.pth.tar'
checkpoint = torch.load(transfer, map_location=device)
model = Roost(**checkpoint["model_params"], device=device,)
model.to(device)
model.load_state_dict(checkpoint["state_dict"])
fea_path = '../roost/data/el-embeddings/matscholar-embedding.json'

def gen_feats(data_path, task_name):
        task_dict = {task_name:'regression'}
        dataset = CompositionData(data_path=data_path, fea_path=fea_path, task_dict=task_dict)
        n_targets = dataset.n_targets
        elem_emb_len = dataset.elem_emb_len
        generator = DataLoader(dataset, **data_params)
        feats = model.featurise(generator)
        formation_energies = model.predict(generator)
        return feats, formation_energies

def return_function(filename='mp_data.csv'):
        mp_feats, formation_energies = gen_feats(filename, task_name='PFn_max')
        return mp_feats, formation_energies

def formation_energies(data_path, task_name):
        task_dict = {task_name:'regression'}
        dataset = CompositionData(data_path=data_path, fea_path=fea_path, task_dict=task_dict)
        n_targets = dataset.n_targets
        elem_emb_len = dataset.elem_emb_len
        generator = DataLoader(dataset, **data_params)
        feats = model.predict(generator)
        return feats

