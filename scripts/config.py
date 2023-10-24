from os.path import join
from pathlib import Path


SAMPLE_RATE = 100

FOOOF_PARAMS = {"max_n_peaks": 1, "peak_width_limits": (4, 8), "verbose": True}
FIT_RANGE = [1, 50]

# if directory does not exist, create it
PLOT_PATH = join("results", "plots")
Path(PLOT_PATH).mkdir(exist_ok=True)

BANDS = {"theta": [4, 8], "alpha": [8, 12], "beta": [13, 35],
         "gamma": [35, 50]}