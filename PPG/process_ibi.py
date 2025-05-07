import heartpy as hp
import numpy as np
import pandas as pd
from scipy import signal, fftpack
from scipy.stats import iqr
from scipy.signal import spectrogram
import bisect
import matplotlib.pyplot as plt

def find_closest_index(a, x):
    i = bisect.bisect_left(a, x)
    if i >= len(a):
        i = len(a) - 1
    elif i and a[i] - x > x - a[i - 1]:
        i = i - 1
    return i

def peak_interp(yn, yn_1, yn_2, time, delta):
    peak_position = [0, 0]
    alph = yn_2
    beta = yn_1
    gam = yn
    pp = 0.5 * (alph - gam) / (alph - 2 * beta + gam)
    peak_position[1] = beta - 0.25 * (alph - gam) * pp
    peak_position[0] = time + pp * delta
    return peak_position


def peak_det(ppg_raw, ts, sample_rate):
    # fs = len(ppg_raw) / (ts[-1] - ts[0])
    fs = sample_rate
    ppg_filt = ppg_raw
    num_samp = len(ppg_filt)
    peaks, troughs = [], []

    dx = ppg_filt[1] - ppg_filt[0]
    peak_search = True
    for ii in range(2, num_samp):
        dxp = dx
        dx = ppg_filt[ii] - ppg_filt[ii - 1]
        if np.sign(dx) != np.sign(dxp):
            if np.sign(dxp) == 1:
                peak_position = peak_interp(ppg_filt[ii], ppg_filt[ii - 1], ppg_filt[ii - 2], ts[ii - 1],
                                                    1 / fs)
                if peak_position[1] > 0:
                    if not peak_search:
                        cval = peak_position[1]
                        pval = peaks[-1][1]
                        if cval > pval:
                            peaks[-1] = peak_position
                    else:
                        peak_search = False
                        peaks.append(peak_position)
            else:
                peak_position = peak_interp(ppg_filt[ii], ppg_filt[ii - 1], ppg_filt[ii - 2], ts[ii - 1], 1 / fs)
                if peak_position[1] < 0:
                    if (peak_search == True) and (len(troughs) > 0):
                        cval = peak_position[1]
                        pval = troughs[-1][1]
                        if cval < pval:
                            troughs[-1] = peak_position
                    else:
                        peak_search = True
                        troughs.append(peak_position)
    return np.array(peaks), np.array(troughs)


def peak_outliers(peaks, IBI_threshold=(0.2, 2)):
    if len(peaks) <= 1:  # no IBI detected
        return None, None
    IBI_va = np.diff(peaks[:, 0])
    IBI_ts = peaks[1:, 0]
    IBI_ori = pd.DataFrame({'ts': IBI_ts, 'val': IBI_va})
    qualityGood, qualityPoor = 0, 1
    outlier = np.ones(len(IBI_ori))
    valid_rrInterval = IBI_ori

    # single_ibi_histogram(IBI_va*1000, 'Raw Earbud IBI')

    diff_rrInterval = np.abs(np.diff(valid_rrInterval['val']))
    MED = 3.32 * iqr(diff_rrInterval)
    MAD = (np.median(valid_rrInterval['val']) - 2.9 * iqr(diff_rrInterval)) / 3
    CBD = (MED + MAD) / 2
    # if CBD < 0.2:
    #     CBD = 0.2

    outlier[0] = qualityGood
    standard_rrInterval = valid_rrInterval['val'].iloc[0]
    prev_beat_bad = 0
    for i in range(1, len(valid_rrInterval) - 1):
        if (valid_rrInterval['val'].iloc[i] > IBI_threshold[0]) & (
                valid_rrInterval['val'].iloc[i] < IBI_threshold[1]):
            beat_diff_prevGood = np.abs(standard_rrInterval - valid_rrInterval['val'].iloc[i])
            beat_diff_prev = np.abs(valid_rrInterval['val'].iloc[i - 1] - valid_rrInterval['val'].iloc[i])
            beat_diff_post = np.abs(valid_rrInterval['val'].iloc[i] - valid_rrInterval['val'].iloc[i + 1])
            if (prev_beat_bad == 1) & (beat_diff_prevGood < CBD):
                outlier[i] = qualityGood
                prev_beat_bad = 0
                standard_rrInterval = valid_rrInterval['val'].iloc[i]
            elif (prev_beat_bad == 1) & (beat_diff_prevGood > CBD) & (beat_diff_prev <= CBD):
                outlier[i] = qualityGood
                prev_beat_bad = 0
                standard_rrInterval = valid_rrInterval['val'].iloc[i]
            elif (prev_beat_bad == 1) & (beat_diff_prevGood > CBD) & (
                    (beat_diff_prev > CBD) | (beat_diff_post > CBD)):
                prev_beat_bad = 1
            elif (prev_beat_bad == 0) & (beat_diff_prev <= CBD):
                outlier[i] = qualityGood
                prev_beat_bad = 0
                standard_rrInterval = valid_rrInterval['val'].iloc[i]
            elif (prev_beat_bad == 0) & (beat_diff_prev > CBD):
                prev_beat_bad = 1

    peaks_ind_outlier = [ind + 1 for (ind, flag) in enumerate(outlier) if flag > 0]
    peaks_ind_good = [ind for ind in range(len(peaks)) if ind not in peaks_ind_outlier]

    return peaks_ind_outlier, peaks_ind_good


def peak_det2(ppg_raw, ts, ibi_threshold=(0.4, 1.5), fs=25):
    peaks, _ = peak_det(ppg_raw, ts, fs)
    if len(peaks) <= 1:  # No IBI detected
        return [], [], [], [], []
    peaks_ind_outlier, peaks_ind_ppg = peak_outliers(peaks, IBI_threshold=ibi_threshold)

    ibi_orig = np.diff(peaks[:, 0])
    ibi_ts = peaks[1:, 0]
    # remove ibi associated to 1) peak outliers, and 2) the peak immediately after peak outliers
    ibi_outlier_ind = [x - 1 for x in peaks_ind_outlier] + [x for x in peaks_ind_outlier if x != len(peaks) - 1]

    outlier_ibi = ibi_orig[ibi_outlier_ind]
    ibi = np.delete(ibi_orig, ibi_outlier_ind)
    # ibi = ibi_orig

    outlier_ts = ibi_ts[ibi_outlier_ind]
    ibi_ts = np.delete(ibi_ts, ibi_outlier_ind)

    ibi_ind = []

    for t in ibi_ts:
        ibi_ind.append(find_closest_index(ts, t))

    outlier_ind = []

    for t in outlier_ts:
        outlier_ind.append(find_closest_index(ts, t))

    return peaks, peaks_ind_outlier, peaks_ind_ppg, ibi, ibi_ts


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    b, a = signal.butter(order, [low, high], btype='bandpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def get_ibi(input_signal, time_start, sample_rate=25):
    wd, m = hp.process(input_signal, sample_rate=sample_rate)

    df_ibi = pd.DataFrame({"rr_list": wd.get("RR_list"), "rr_valid_flag": 1 - np.array(wd.get("RR_masklist"))})
    df_ibi['timestamp'] = wd.get("peaklist")[1:]
    df_ibi['timestamp'] = ((df_ibi['timestamp'] / sample_rate)*1000) + time_start

    df_ibi = df_ibi[df_ibi["rr_list"] < 3000]

    # plotPeaks(input_signal, wd.get("peaklist"), 'Peaks')

    return df_ibi


def get_ibi_max_slope(input_signal, time_start, sample_rate=25):

    diff_signal = np.diff(input_signal)

    wd, m = hp.process(diff_signal, sample_rate=sample_rate)

    df_ibi = pd.DataFrame({"rr_list": wd.get("RR_list"), "rr_valid_flag": 1 - np.array(wd.get("RR_masklist"))})

    peaks_array = np.asarray(wd.get("peaklist")[1:])
    adjusted_peaks = peaks_array + 1
    df_ibi['timestamp'] = ((adjusted_peaks / sample_rate)*1000) + time_start

    df_ibi = df_ibi[df_ibi["rr_list"] < 3000]

    return df_ibi


def compute_snr(ppg, ifdetrend4snr=1, fs=100, bound_hr=(0.6, 3), low=0.5, high=8, ifplot=0):
    # snr_win_size: the window length, in minute, to calculate snr

    ppg_filt = butter_bandpass_filter(ppg, low, high, fs=fs, order=3)

    if ifdetrend4snr == 1:
        sig_filt = signal.detrend(ppg_filt)

    N = len(sig_filt)

    # normalize signal between 0 and 1
    sig_filt_norm = (sig_filt - np.min(sig_filt)) / (np.max(sig_filt) - np.min(sig_filt))

    mag_fft = fftpack.fft(sig_filt_norm)
    mag_fft = 2.0 / N * np.abs(mag_fft[:N // 2])
    bin_fft = np.linspace(0.0, 1.0 // (2.0 / fs), len(sig_filt_norm) // 2)

    low_bound_hr, high_bound_hr = bound_hr[0], bound_hr[1]
    # low_bound_range, high_bound_range = np.max([0.5, low_bound_hr]), np.min([high_bound_hr, bin_fft[-1]])

    mag_fft_for_peak = mag_fft[(bin_fft > low_bound_hr) & (bin_fft < high_bound_hr)]
    bin_fft_for_peak = bin_fft[(bin_fft > low_bound_hr) & (bin_fft < high_bound_hr)]

    peak_idx = np.argmax(mag_fft_for_peak)

    f_peak = bin_fft_for_peak[peak_idx]

    harmonics2_peak, harmonics3_peak = 2 * f_peak, 3 * f_peak

    def fre2ind(fre, bin_fft_infun=bin_fft):
        return np.argmin(np.abs(fre - bin_fft_infun))

    power_sig = np.sum(mag_fft[fre2ind(f_peak - 0.1): fre2ind(f_peak + 0.1)] ** 2) + \
                np.sum(mag_fft[fre2ind(harmonics2_peak - 0.1): fre2ind(harmonics2_peak + 0.1)] ** 2) + \
                np.sum(mag_fft[fre2ind(harmonics3_peak - 0.1): fre2ind(harmonics3_peak + 0.1)] ** 2)
    power_noi = np.sum(mag_fft[: fre2ind(harmonics3_peak + 0.1)] ** 2) - power_sig

    snr = 10 * np.log10(power_sig / power_noi)

    if ifplot:
        plt.figure(figsize=(5, 3))
        plt.plot(bin_fft[1:], mag_fft[1:])
        plt.axvline(f_peak - 0.1, color='r')
        plt.axvline(f_peak + 0.1, color='r')
        plt.axvline(harmonics2_peak - 0.1, color='r')
        plt.axvline(harmonics2_peak + 0.1, color='r')
        plt.axvline(harmonics3_peak - 0.1, color='r')
        plt.axvline(harmonics3_peak + 0.1, color='r')
        plt.title(f'PPG, SNR={round(snr, 2)}')
        plt.xlim((0, 5))
        plt.show()
    return snr

def spectral_entropy(Sxx, eps=1e-12):
    power_time = np.sum(Sxx, axis=0, keepdims=True)
    p = Sxx / (power_time + eps)
    entropy_per_frame = -np.sum(p*np.log2(p + eps), axis=0)
    return np.mean(entropy_per_frame)

def compute_entropy(seg, threshold=2):
    nperseg = 512
    pad_size = nperseg // 2
    x_padded = np.pad(seg, (pad_size, pad_size), mode="constant")
    f, time_segments, Sxx = spectrogram(x_padded, fs=100, nperseg=nperseg, noverlap=pad_size)
    mean_entropy = spectral_entropy(Sxx)
    return mean_entropy