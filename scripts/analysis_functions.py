from scipy.stats import spearmanr
import numpy as np
import scipy.signal as sig
import seaborn as sns
import matplotlib.pyplot as plt
from specparam import SpectralModel, SpectralGroupModel
from specparam.analysis import get_band_peak_group
from scripts.utils import _load_data, _load_lfp, _load_spiking
import scripts.config as cfg
from os.path import join


def correlations(fname="steinmetz_2016-12-14_Cori.nc", brain_area="MOs"):

    data = _load_data(fname)

    lfp = _load_lfp(data, brain_area)
    spiking_mean = _load_spiking(data, brain_area, area_mean=True,
                                 trial_sum=True)

    # Get power spectral density of LFP
    sample_rate = cfg.SAMPLE_RATE
    freqs, psd = sig.welch(lfp, fs=sample_rate, nperseg=sample_rate // 2)
    median_psd = np.median(psd, axis=1)

    fg = fit_fooof(freqs, psd)
    offsets = fg.get_params('aperiodic_params', "offset")
    exponents = fg.get_params('aperiodic_params', "exponent")
    aperiodic_powers = get_aperiodic_powers(freqs, psd, log=True)
    theta_power = band_power(fg, "theta")
    alpha_power = band_power(fg, "alpha")

    plot_correlation(spiking_mean, median_psd, "Median Power Spectral Density")
    plot_correlation(spiking_mean, offsets, "PSD Offset")
    plot_correlation(spiking_mean, exponents, "PSD Exponent")
    plot_correlation(spiking_mean, aperiodic_powers, "Mean Aperiodic Power")
    plot_correlation(spiking_mean, theta_power, "Theta Power")
    plot_correlation(spiking_mean, alpha_power, "Alpha Power")


def band_power(fg, band):
    pwr_idx = 1
    freq_range = cfg.BANDS[band]
    band_power = get_band_peak_group(fg, freq_range)[:, pwr_idx]
    band_power[np.isnan(band_power)] = 0  # replace nan with 0
    return band_power


def fit_fooof(freqs, psd):
    fg = SpectralGroupModel(**cfg.FOOOF_PARAMS)
    fg.fit(freqs, psd, freq_range=cfg.FIT_RANGE)
    fg.plot(save_fig=True, file_name="fooof_results.pdf",
            file_path=cfg.PLOT_PATH)
    return fg


def get_aperiodic_powers(freqs, psd, log=False):
    fm = SpectralModel(**cfg.FOOOF_PARAMS)
    mean_aperiodic_powers = []
    for trial_pwr in psd:
        fm.fit(freqs, trial_pwr, freq_range=cfg.FIT_RANGE)
        offset, exponent = fm.get_params('aperiodic_params')
        aperiodic_power = 10**offset * freqs[1:]**(-exponent)
        # mean_ap = aperiodic_power.mean()
        mean_aperiodic_powers.append(aperiodic_power.mean())
    mean_aperiodic_powers = np.array(mean_aperiodic_powers)
    if log:
        mean_aperiodic_powers = np.log(mean_aperiodic_powers)
    return mean_aperiodic_powers


def plot_correlation(x, y, y_label, x_label="Spikes per epoch"):
    rho, pval = spearmanr(x, y)

    fig, ax = plt.subplots()
    sns.regplot(x=x, y=y, label=f"rho={rho:.2f}, p={pval:.2f}", ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    fname = join(cfg.PLOT_PATH, f"{y_label}.pdf")
    plt.savefig(fname)


if __name__ == '__main__':
    correlations()