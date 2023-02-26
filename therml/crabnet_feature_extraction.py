import os
import numpy as np
import pandas as pd
import torch
import pickle

from sklearn.metrics import roc_auc_score

from crabnet.kingcrab import CrabNet, tCrabNet
from crabnet.model import Model
from utils.get_compute_device import get_compute_device

def get_features():

	mat_prop = 'power_factor'
	transfer = 'aflow__agl_thermal_conductivity_300K'#'OQMD_Formation_Enthalpy'
	compute_device = get_compute_device(prefer_last=True)
	RNG_SEED = 42
	torch.manual_seed(RNG_SEED)
	np.random.seed(RNG_SEED)

	model = Model(tCrabNet(compute_device=compute_device).to(compute_device),\
			model_name=f'{mat_prop}', verbose=True)

	model.load_network(f'{transfer}.pth')

	model.load_data('/scratch/mxyptlkr/CrabNet/data/power_factor/val.csv', batch_size=32, train=False)
	output = np.array(model.predict(model.data_loader,feature_extraction=True)[1])
	print(output.shape)

	return output

