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
    spike_plot_by_trials(lfp, neural_spiking, "alpha")
    spike_plot_by_trials(lfp, neural_spiking, "theta")


def spike_plot(lfp, neural_spiking, band):
    # pooled_neurons = neural_spiking.sum(0)
    # concatenate all 364 trials together
    spike_lfp_phases = _get_spike_lfp_phases(lfp, neural_spiking, band)
    # Create a polar histogram to visualize the spike phase distribution.
    # The `bins` parameter controls the number of bins in the polar plot.

    plt.figure(figsize=(8, 8))
    n, bins, _ = plt.hist(spike_lfp_phases, bins=36, density=True,
                          color='b', alpha=0.7)
    plt.clf()  # Clear the histogram from the current figure

    ax = plt.subplot(111, projection='polar')
    # Set the top of the plot as North (0 radians)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)      # Rotate clockwise

    # Normalize the bin heights so that the integral is 1 (density plot)
    width = 2 * np.pi / len(bins)
    ax.bar(bins[:-1], n, width=width, align="edge", edgecolor='k')

    plt.title("Spike-LFP Phase Distribution Alpha band \n Number of spikes: "
              "{sum(spiking_activity_global_concatenate)}\n p_value= "
              "{p_value}", va='bottom')
    save_path = join(cfg.PLOT_PATH, f"spike_lfp_phase_distribution_{band}.pdf")
    plt.savefig(save_path)


def spike_plot_by_trials(lfp, neural_spiking, band):
    significant_trials_band = []
    lowcut, highcut = cfg.BANDS[band]

    for trial_idx in range(0, neural_spiking.shape[1], 1):
        lfp_activity_trial_i = lfp[trial_idx]
        spiking_activity_trial_i = neural_spiking[:, trial_idx, :].sum(0)

        # Apply the bandpass filter to your LFP data.
        filtered_lfp = _bandpass_filter(lfp_activity_trial_i, lowcut, highcut,
                                        100)

        analytic_signal = hilbert(filtered_lfp)
        phase = np.angle(analytic_signal)

        spike_idx = np.where(spiking_activity_trial_i != 0)[0]

        # Extract LFP phase values at spike times
        spike_lfp_phases = phase[spike_idx]

        # Calculate the mean phase angle
        mean_phase = np.angle(np.mean(np.exp(1j * spike_lfp_phases)))

        # Perform a Rayleigh test
        result = pg.circ_rayleigh(spike_lfp_phases)

        # Extract the test statistic and p-value
        # test_statistic = result[0]
        p_value = result[1]

        if p_value < 0.05:
            print("Significant phase locking detected.")
            significant_trials_band.append(trial_idx)
            plt.figure()
            # Create a polar histogram to visualize the spike phase
            # distribution. The `bins` parameter controls the number of bins in
            # the polar plot.

            plt.figure(figsize=(8, 8))
            n, bins, patches = plt.hist(spike_lfp_phases, bins=36,
                                        density=True, color='b', alpha=0.7)
            plt.clf()  # Clear the histogram from the current figure

            # Create a polar plot with phase values on the x-axis and spike
            # density (or count) on the y-axis.

            ax = plt.subplot(111, projection='polar')
            # Set the top of the plot as North (0 radians)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)      # Rotate clockwise

            # Normalize the bin heights so that the integral is 1
            # (density plot)
            width = 2 * np.pi / len(bins)
            _ = ax.bar(bins[:-1], n, width=width, align="edge",
                       edgecolor='k')

            # Normalize the arrow length based on the mean phase
            arrow_length = np.abs(np.mean(np.exp(1j * spike_lfp_phases)))

            # Add the arrow
            ax.arrow(mean_phase, 0, arrow_length, 0, head_width=0.1,
                     head_length=0.1, fc='red', ec='red')

            plt.title("Spike-LFP Phase Distribution Alpha band \n\n Trial # "
                      f"{trial_idx} \n Number of spikes: "
                      f"{sum(spiking_activity_trial_i)}\n Number of neurons: "
                      f"{neural_spiking.shape[0]} \np_value= {p_value}",
                      va='bottom')
            fig_name = (f"spike_lfp_phase_distribution_trial_{trial_idx}_"
                        f"{band}.pdf")
            plt.savefig(join(cfg.PLOT_PATH, fig_name))
    print(f"significant phase-locking with {band} in trials #: "
          f"{significant_trials_band}")


def _get_spike_lfp_phases(lfp, neural_spiking, band, trial_idx=None):
    if isinstance(trial_idx, int):
        mask = slice(trial_idx)
        lfp_masked = lfp[mask]
    else:
        lfp_masked = np.hstack(lfp)
    # lfp_activity_trial_i = lfp[trial_idx]
    lowcut, highcut = cfg.BANDS[band]
    # Apply the bandpass filter to your LFP data.
    filtered_lfp = _bandpass_filter(lfp_masked, lowcut, highcut, 100)
    analytic_signal = hilbert(filtered_lfp)
    phase = np.angle(analytic_signal)
    if isinstance(trial_idx, int):
        spiking_activity_trial_i = neural_spiking[:, trial_idx, :].sum(0)
        spike_idx = np.where(spiking_activity_trial_i != 0)[0]
    else:
        pooled_neurons = neural_spiking.sum(0)
        # concatenate all 364 trials together
        spiking_activity_global_concatenate = np.hstack(pooled_neurons)
        spike_idx = np.where(spiking_activity_global_concatenate != 0)[0]
    # Extract LFP phase values at spike times
    spike_lfp_phases = phase[spike_idx]
    return spike_lfp_phases


def _bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


if __name__ == "__main__":
    phase_spiking()