import heartpy as hp
import numpy as np
from scipy import signal


# Orson code used 0.7 to 3.5Hz bandpass, 3rd order
# AFib paper used 0.6 to 3.2Hz bandpass, FIR filter
def filter_ppg(ppg_signal, sampling_rate=25.0, band=[0.5, 2.8]):
    return hp.filter_signal(ppg_signal, band, sample_rate=sampling_rate, order=3, filtertype='bandpass')


def fine_filter_ppg(ppg_signal, fft_hr, fine_range=0.2, sampling_rate=25.0, order=3):
    center_freq = fft_hr / 60
    filter_low_fine, filter_high_fine = center_freq - fine_range, center_freq + fine_range

    nyq = sampling_rate * 0.5

    b, a = signal.butter(order, [filter_low_fine / nyq, filter_high_fine / nyq], btype='bandpass')
    ppg_fine = signal.filtfilt(b, a, ppg_signal, method='gust')

    return ppg_fine


def get_hr_fft(signal, low_hr, high_hr, sampling_rate=25.0):
    raw = np.abs(np.fft.rfft(signal))  # do real fft
    raw /= np.max(raw)
    L = signal.size

    extended_raw = np.abs(np.fft.rfft(signal, n=10 * L))  # do real fft with zero padding
    extended_raw /= np.max(extended_raw)
    freqs = float(sampling_rate) / L * np.arange(np.floor(L / 2) + 1)
    extended_freqs = float(sampling_rate) / L * np.arange(start=0, stop=np.floor(L / 2) + 1. / 10,
                                                          step=1. / 10)

    bpms = 60. * extended_freqs
    range_idx = np.where((bpms >= low_hr) & (bpms <= high_hr))  # the range of vital sign is supposed to be within
    range_fft = extended_raw[range_idx]
    range_fft /= np.max(range_fft)
    range_bpms = bpms[range_idx]
    range_freqs = extended_freqs[range_idx]

    peak_idx = np.argmax(range_fft)  # [np.argmax(properties['prominences'])]
    bpm = range_bpms[peak_idx]

    return bpm