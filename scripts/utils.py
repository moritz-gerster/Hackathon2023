import os

import netCDF4 as nc
import numpy as np


def _load_data(fname):
    source_path = "sourcedata"
    data_path = os.path.join(source_path, fname)
    data = nc.Dataset(data_path)
    return data


def _load_lfp(data, brain_area):
    brain_areas = _get_brain_areas(data)
    brain_area_idx = np.where(brain_areas == brain_area)[0][0]
    lfp_data = data.variables["lfp"]
    brain_area_lfp = lfp_data[brain_area_idx].data
    return brain_area_lfp


def _load_spiking(data, brain_area):
    brain_areas = data.variables["brain_area"][:]
    brain_area_idx = np.where(brain_areas == brain_area)[0]
    spiking_data = data.variables["spike_rate"]
    neural_spiking = spiking_data[brain_area_idx].data
    return neural_spiking


def _get_brain_areas(data):
    brain_areas = data.variables["brain_area_lfp"][:]
    return brain_areas