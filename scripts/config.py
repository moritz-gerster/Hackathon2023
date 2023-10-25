from os.path import join


SAMPLE_RATE = 100

FOOOF_PARAMS = {"max_n_peaks": 1, "peak_width_limits": (4, 8), "verbose": True}
FIT_RANGE = [1, 50]

PLOT_PATH = join("results", "plots")

BANDS = {"theta": [4, 8], "alpha": [8, 12], "beta": [13, 35],
         "gamma": [35, 50]}