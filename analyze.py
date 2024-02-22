import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import scipy.optimize
from read_waveform import data_reader

matplotlib.use("WebAgg")


def average(xs):
    return xs.sum / xs.size


def standard_deviation(xs):
    return (average(xs**2) - average(xs) ** 2) ** (1 / 2)


def get_average_waveform(waveforms):
    return np.mean(waveforms, axis=0)


def get_baseline(baseline_waveform, iterations=5):
    if iterations == 0:
        return np.mean(baseline_waveform)
    avg = np.mean(baseline_waveform)
    std = np.std(baseline_waveform)
    return get_baseline(
        baseline_waveform[baseline_waveform < avg + 3 * std], iterations - 1
    )


def get_baselines(baseline_waveforms):
    return np.apply_along_axis(get_baseline, 1, baseline_waveforms)


def get_baseline_subtracted_waveforms(waveforms, baselines):
    return waveforms - baselines[:, np.newaxis]


def get_charges(pulses):
    return pulses.sum(axis=1)


def get_pulse_start(average_waveform, divisions=100):
    window = average_waveform.size // divisions
    mean = 0
    std = 0
    for i in range(average_waveform.size // 10, average_waveform.size, window):
        mean = np.mean(average_waveform[:i])
        std = np.std(average_waveform[:i])
        if np.mean(average_waveform[i : i + window]) - mean > 4 * std:
            return i


def get_pulse_end(average_waveform, average_baseline, std_baseline, divisions=100):
    window = average_waveform.size // divisions
    length = average_waveform.size
    for i in reversed(range(0, length - window, window)):
        if (
            np.mean(average_waveform[i : i + window]) - average_baseline
            > 3 * std_baseline
        ):
            return i


def calculate_waveform_parameters(waveforms):
    average_waveform = get_average_waveform(waveforms)
    pulse_start = get_pulse_start(average_waveform)
    average_baseline = np.mean(average_waveform[:pulse_start])
    std_baseline = np.std(average_waveform[:pulse_start])
    pulse_end = get_pulse_end(average_waveform, average_baseline, std_baseline)
    plt.plot(average_waveform)
    plt.axvline(x=pulse_start, color="k", label="Pulse start")
    plt.axvline(x=pulse_end, color="k", label="Pulse end")
    print("pulse_start =", pulse_start)
    print("pulse_end =", pulse_end)


def calculate_baseline_cut(baselines):
    baseline_cut = np.nanmean(baselines) + 3 * np.std(baselines)
    print("baseline cut =", baseline_cut)
    return baseline_cut


def gaussian(x, H, A, x0, sigma, c):
    return H + A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def get_histogram(data):
    num_bins = 100
    x_max = 20000
    hist = np.histogram(data, bins=num_bins, range=(0, x_max))
    xs = hist[1][:-1] + x_max / num_bins / 2
    ys = hist[0]
    return xs, ys


def get_multipeak_fit(charges, offset, interval, num_peaks):
    half_width = interval / 3
    xs, ys = get_histogram(charges)
    scipy.optimize.curve_fit(
        gaussian,
        xs[offset - half_width <= xs < offset + half_width],
        ys[offset - half_width <= xs < offset + half_width],
    )
    for i in range(num_peaks):
        print(i)

    return 0


waveforms = np.array(
    data_reader(
        "data/20240115_SPE_LED365nm/SPE_365nm/run16_C1_LED_20ns_3V30/0_wave0_C1_LED_20ns_3V30.dat",
        10000,
        100,
    )
)

# calculate_waveform_parameters(waveforms)
pulse_start = 4100
pulse_end = 6000
# baseline_cut = 20

baselines = get_baselines(waveforms[:, :pulse_start])
baseline_subtracted_waveforms = get_baseline_subtracted_waveforms(waveforms, baselines)
baseline_cut = calculate_baseline_cut(baselines)
cut_waveforms = baseline_subtracted_waveforms[baselines < baseline_cut]
charges = get_charges(cut_waveforms[:, pulse_start:pulse_end])

plt.plot(np.transpose(baseline_subtracted_waveforms[50:65, pulse_start:pulse_end]))
# plt.plot(get_average_waveform(baseline_subtracted_waveforms))
# plt.hist(baselines)
# plt.hist(charges, bins=100, range=(0, 20000))
plt.show()

