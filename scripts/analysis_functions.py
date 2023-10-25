from os.path import join
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sig
import seaborn as sns
from scipy.stats import spearmanr
from specparam import SpectralGroupModel, SpectralModel
from specparam.analysis import get_band_peak_group

import scripts.config as cfg
from scripts.utils import _load_data, _load_lfp, _load_spiking


def correlations(fname="steinmetz_2016-12-14_Cori.nc", brain_area="MOs"):
    data = _load_data(fname)
    lfp = _load_lfp(data, brain_area)
    neural_spiking = _load_spiking(data, brain_area)

    # neural_spiking has shape (n_neurons, n_trials, n_timepoints)
    # -> average across all neurons to get spiking over entire brain area
    spiking_mean = neural_spiking.mean(0)
    # take sum across timepoints to get spiking per trial
    spiking_mean_sum = spiking_mean.sum(-1)
    spiking = spiking_mean_sum

    # Get power spectral density of LFP
    sample_rate = cfg.SAMPLE_RATE
    freqs, psd = sig.welch(lfp, fs=sample_rate, nperseg=sample_rate // 2)
    median_psd = np.median(psd, axis=1)

    save_path = join(cfg.PLOT_PATH, brain_area)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    fg = fit_fooof(freqs, psd, save_path=save_path)
    offsets = fg.get_params('aperiodic_params', "offset")
    exponents = fg.get_params('aperiodic_params', "exponent")
    aperiodic_powers = get_aperiodic_powers(freqs, psd)
    aperiodic_powers_mean_log = np.log10(aperiodic_powers).mean(1)
    # theta_power = band_power(fg, "theta")
    # alpha_power = band_power(fg, "alpha")

    plot_correlation(spiking, median_psd, "Median Power Spectral Density",
                     save_path=save_path)
    plot_correlation(spiking, offsets, "PSD Offset", save_path=save_path)
    plot_correlation(spiking, exponents, "PSD Exponent", save_path=save_path)
    plot_correlation(spiking, aperiodic_powers_mean_log,
                     "Mean Aperiodic Log Power", save_path=save_path)
    # plot_correlation(spiking, theta_power, "Theta Power",
    #                  save_path=save_path)
    # plot_correlation(spiking, alpha_power, "Alpha Power",
    #                  save_path=save_path)

    plot_example(freqs, psd, aperiodic_powers, neural_spiking,
                 save_path=save_path)


def band_power(fg, band):
    pwr_idx = 1
    freq_range = cfg.BANDS[band]
    band_power = get_band_peak_group(fg, freq_range)[:, pwr_idx]
    band_power[np.isnan(band_power)] = 0  # replace nan with 0
    return band_power


def fit_fooof(freqs, psd, save_path=None):
    fg = SpectralGroupModel(**cfg.FOOOF_PARAMS)
    fg.fit(freqs, psd, freq_range=cfg.FIT_RANGE)
    save_fig = True if save_path is not None else False
    fg.plot(save_fig=save_fig, file_name="fooof_results.pdf",
            file_path=save_path)
    return fg


def get_aperiodic_powers(freqs, psd, mean=False):
    fm = SpectralModel(**cfg.FOOOF_PARAMS)
    mean_aperiodic_powers = []
    for trial_pwr in psd:
        fm.fit(freqs, trial_pwr, freq_range=cfg.FIT_RANGE)
        offset, exponent = fm.get_params('aperiodic_params')
        aperiodic_power = 10**offset * freqs[1:]**(-exponent)
        mean_aperiodic_powers.append(aperiodic_power)
    mean_aperiodic_powers = np.array(mean_aperiodic_powers)
    return mean_aperiodic_powers


def plot_correlation(x, y, y_label, x_label="Spikes per epoch", save_path=None):
    rho, pval = spearmanr(x, y)

    _, ax = plt.subplots()
    sns.regplot(x=x, y=y, label=f"rho={rho:.2f}, p={pval:.2f}", ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(save_path.split("/")[-1])
    ax.legend()
    if save_path:
        fname = join(save_path, f"{y_label}.pdf")
        plt.savefig(fname)


def plot_example(freqs, psd, aperiodic_powers, neural_spiking, n_trials=30,
                 save_path=None):

    argmax_neuron = _get_best_neuron(psd, neural_spiking)
    best_neuron = neural_spiking[argmax_neuron]
    mean_firing_rate = _get_firing_rate(best_neuron)
    argmax_trial = _get_best_trial(psd, best_neuron, n_trials)
    fontsize = 15

    time_bins = neural_spiking.shape[-1]
    freq_bins = aperiodic_powers.shape[-1]
    ones_freqs = np.ones(freq_bins)
    ones_time = np.ones(time_bins)

    gridspec_kw = {"wspace": 0, "hspace": .4, "height_ratios": [1, .2, 1]}
    _, axes = plt.subplots(3, n_trials, figsize=(20, 10), sharey="row",
                           sharex=False, gridspec_kw=gridspec_kw)
    for i, trial in enumerate(range(argmax_trial, argmax_trial + n_trials)):
        axes[0, i].loglog(freqs, psd[trial], "k")
        axes[0, i].loglog(freqs[1:], aperiodic_powers[trial], "darkorange")
        trial_mean = ones_freqs * aperiodic_powers[trial].mean()
        axes[0, i].loglog(freqs[1:], trial_mean, "r")
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[0, i].set_xlim(freqs[0], freqs[-1])

        axes[1, i].plot(best_neuron[trial], "k")
        axes[1, i].set_yticks([])
        axes[1, i].set_xticks([])

        axes[2, i].plot(mean_firing_rate[trial], "k")
        mean_firing_rate_avg = ones_time * mean_firing_rate[trial].mean()
        axes[2, i].plot(mean_firing_rate_avg, "b")
        # mean_ap_powers_scaled = mean_ap_powers[trial] / mean_ap_powers.max()
        # mean_ap_powers_scaled *= np.nanmax(mean_firing_rate)
        # mean_ap_powers_scaled = ones * mean_ap_powers_scaled.mean()
        # axes[2, i].plot(mean_ap_powers_scaled, "darkorange")
        axes[2, i].set_xticks([])

    axes[0, 0].set_xticks([1, 10, 50], labels=[1, 10, 50], minor=False)
    axes[0, 0].set_xlabel("Frequency (Hz)")
    axes[0, 0].set_ylabel("Power Spectral Density", fontsize=fontsize)

    axes[1, 0].set_ylabel("Spikes")
    axes[1, 0].set_yticks([0, 2, 4])

    axes[2, 0].set_ylabel("Firing Rate (Hz)", fontsize=fontsize)
    x_middle = n_trials // 2
    axes[2, x_middle].set_xlabel("Time (s)", fontsize=fontsize, labelpad=20)
    [axes[2, j].set_xlabel(f"{j/4:.0f}", fontsize=fontsize)
     for j in range(0, n_trials, 4)]
    brain_area = save_path.split("/")[-1]
    plt.suptitle(brain_area, fontsize=fontsize*2)
    if save_path:
        plt.savefig(join(save_path, f"{brain_area}.pdf"))


def _get_best_neuron(psd, neural_spiking):
    rhos = []
    pvals = []
    mean_psd = psd.mean(1)
    for neuron in range(neural_spiking.shape[0]):
        single_neuron_spike_rate = neural_spiking[neuron].mean(-1)
        rho, pval = spearmanr(mean_psd, single_neuron_spike_rate)
        rhos.append(rho)
        pvals.append(pval)
    rhos = np.array(rhos)
    pvals = np.array(pvals)
    argmax = np.argmax(rhos)
    print(f"Best neuron idx={argmax}: rho={rhos[argmax]:.2f}, "
          "p={pvals[argmax]:.2f}")
    return argmax


def _get_best_trial(psd, best_neuron, n_trials):
    rhos = []
    pvals = []
    mean_psd = psd.mean(1)
    for trial_start in range(best_neuron.shape[0] - n_trials):
        mask = slice(trial_start, trial_start + n_trials)
        single_neuron_spike_rate = best_neuron[mask].mean(-1)
        rho, pval = spearmanr(mean_psd[mask], single_neuron_spike_rate)
        rhos.append(rho)
        pvals.append(pval)
    rhos = np.array(rhos)
    pvals = np.array(pvals)
    argmax = np.nanargmax(rhos)
    print(f"Best trial idx={argmax}: rho={rhos[argmax]:.2f}, "
          "p={pvals[argmax]:.2f}")
    return argmax


def _get_firing_rate(best_neuron):
    original_shape = best_neuron.shape
    series = pd.Series(np.hstack(best_neuron))
    # get rolling average over 10 seconds
    series = series.rolling(cfg.SAMPLE_RATE*10, center=True).sum()
    # convert back into original shape (364, 250)
    mean_firing_rate = np.reshape(series.values, original_shape)
    return mean_firing_rate


if __name__ == '__main__':
    correlations()