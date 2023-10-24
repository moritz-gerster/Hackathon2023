import numpy as np
from pathlib import Path
import os
import netCDF4 as nc


def _load_data(fname):
    source_path = "sourcedata"
    data_path = os.path.join(source_path, fname)
    data = nc.Dataset(data_path)
    return data


def _load_lfp(data, brain_area):
    brain_areas = data.variables["brain_area_lfp"][:]
    brain_area_idx = np.where(brain_areas == brain_area)[0][0]
    lfp_data = data.variables["lfp"]
    brain_area_lfp = lfp_data[brain_area_idx].data
    return brain_area_lfp


def _load_spiking(data, brain_area, area_mean=False, trial_sum=False):
    brain_areas = data.variables["brain_area"][:]
    brain_area_idx = np.where(brain_areas == brain_area)[0]
    spiking_data = data.variables["spike_rate"]
    neural_spiking = spiking_data[brain_area_idx].data
    if area_mean:
        # neural_spiking has shape (n_neurons, n_trials, n_timepoints)
        # -> average across all neurons to get spiking over entire brain area
        neural_spiking = neural_spiking.mean(0)
    if trial_sum:
        neural_spiking = neural_spiking.sum(-1)
    return neural_spiking
