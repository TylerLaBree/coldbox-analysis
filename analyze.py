import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from read_waveform import data_reader


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
    return get_baseline(baseline_waveform[baseline_waveform < avg + 3 * std], iterations-1)


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
    #print(np.shape(baselines))
    #print(baselines)
    baseline_cut = np.nanmean(baselines) + 4 * np.std(baselines) 
    #plt.plot(average_waveform)
    #plt.hlines(baseline_cut, 0, pulse_start, color='k', label='Baseline cut')
    print("baseline_cut =", baseline_cut)


waveforms = np.array(data_reader("data/test.dat", 10000, 200))

# calculate_waveform_parameters(waveforms)
pulse_start = 4100
pulse_end = 6000
baseline_cut = 20

baselines = get_baselines(waveforms[:, :pulse_start])
baseline_subtracted_waveforms = get_baseline_subtracted_waveforms(waveforms, baselines)
calculate_baseline_cut(baselines)
