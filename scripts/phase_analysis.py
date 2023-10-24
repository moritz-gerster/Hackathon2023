from scripts.utils import _load_data, _load_lfp, _load_spiking
import scripts.config as cfg
from scipy.signal import butter, lfilter, hilbert
import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg
from os.path import join


def phase_spiking(fname="steinmetz_2016-12-14_Cori.nc", brain_area="MOs"):

    data = _load_data(fname)
    lfp = _load_lfp(data, brain_area)
    neural_spiking = _load_spiking(data, brain_area)

    spike_plot(lfp, neural_spiking, "alpha")
    spike_plot(lfp, neural_spiking, "theta")


def spike_plot(lfp, neural_spiking, band):
    lowcut, highcut = cfg.BANDS[band]
    # lowcut = 8  # Lower cutoff frequency for alpha (in Hz)
    # highcut = 12  # Upper cutoff frequency for alpha (in Hz)

    # Apply the bandpass filter to your LFP data.
    # concatenate all 364 trials together
    lfp_activity_concatenate = np.hstack(lfp)
    filtered_lfp = _bandpass_filter(lfp_activity_concatenate, lowcut, highcut, 100)

    # calculate phase of oscillations:
    analytic_signal = hilbert(filtered_lfp)
    phase = np.angle(analytic_signal)

    pooled_neurons = neural_spiking.sum(0)
    spiking_activity_global_concatenate = np.hstack(pooled_neurons) # concatenate all 364 trials together
    spike_idx = np.where(spiking_activity_global_concatenate != 0)[0]

    # Extract LFP phase values at spike times
    spike_lfp_phases = phase[spike_idx]

    # Create a polar histogram to visualize the spike phase distribution.
    # The `bins` parameter controls the number of bins in the polar plot.

    plt.figure(figsize=(8, 8))
    n, bins, patches = plt.hist(spike_lfp_phases, bins=36, density=True, color='b', alpha=0.7)
    plt.clf()  # Clear the histogram from the current figure

    # Calculate the mean phase angle
    mean_phase = np.angle(np.mean(np.exp(1j * spike_lfp_phases)))

    # Perform a Rayleigh test
    result = pg.circ_rayleigh(spike_lfp_phases)

    # Extract the test statistic and p-value
    test_statistic = result[0]
    p_value = result[1]

    # Create a polar plot with phase values on the x-axis and spike density (or count) on the y-axis.

    ax = plt.subplot(111, projection='polar')
    ax.set_theta_zero_location("N")  # Set the top of the plot as North (0 radians)
    ax.set_theta_direction(-1)      # Rotate clockwise

    # Normalize the bin heights so that the integral is 1 (density plot)
    width = 2 * np.pi / len(bins)
    bars = ax.bar(bins[:-1], n, width=width, align="edge", edgecolor='k')

    plt.title(f"Spike-LFP Phase Distribution Alpha band \n Number of spikes: {sum(spiking_activity_global_concatenate)}\n p_value= {p_value}", va='bottom')
    save_path = join(cfg.PLOT_PATH, f"spike_lfp_phase_distribution_{band}.pdf")
    plt.savefig(save_path)


def _bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


if __name__ == "__main__":
    phase_spiking()