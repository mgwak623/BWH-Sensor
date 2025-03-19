import pandas as pd
import numpy as np
import pyhrv
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema


def fix_ibi_peaks(ibi_values):
    ibi_fixed = ibi_values
    peaks = argrelextrema(ibi_values, np.greater)
    total_new = 0


    return ibi_fixed


def get_hrv_features(ibi_values):
    column_names = ["ppg_HR", "ppg_LF", "ppg_HF", "ppg_LF_HF_Ratio", "ppg_LF_norm", "ppg_HF_norm",
                    "ppg_VLF", "ppg_TP", "ppg_pNN50", "ppg_NN50", "ppg_RMSSD", "ppg_SDNN", "ppg_TINN", "ppg_SD1"]

    hrv_df = pd.DataFrame(columns=column_names)
    hrv_row = []

    if len(ibi_values) >= 15:

        ibi_mean = np.mean(ibi_values)
        hr_mean = (1 / (ibi_mean / 1000)) * 60

        hrv_tuple = pyhrv.hrv(nni=ibi_values, plot_ecg=False, plot_tachogram=False, show=False)
        plt.close('all')

        hrv_dict = dict(hrv_tuple)

        hrv_row = [hr_mean, hrv_dict['fft_abs'][1], hrv_dict['fft_abs'][2],
                         hrv_dict['fft_ratio'], hrv_dict['fft_norm'][0], hrv_dict['fft_norm'][1],
                         hrv_dict['fft_abs'][0], hrv_dict['fft_total'], hrv_dict['pnn50'],
                         hrv_dict['nn50'], hrv_dict['rmssd'], hrv_dict['sdnn'],
                         hrv_dict['tinn'], hrv_dict['sd1']]

        hrv_df.loc[len(hrv_df)] = hrv_row

        hrv_df['ppg_RSA'] = np.log(hrv_df['ppg_HF'])
    else:
        hrv_row.append([np.nan, np.nan, np.nan,
                         np.nan, np.nan, np.nan,
                         np.nan, np.nan, np.nan,
                         np.nan, np.nan, np.nan,
                         np.nan, np.nan])
        hrv_df = pd.DataFrame(hrv_row, columns=column_names)
        hrv_df['ppg_RSA'] = np.nan

    return hrv_df


def get_segment_features(ibi_values, ibi_times, task_name, subj_name, seg_size):
    column_names = ["Subject", "Task", "Segment", "HR", "RSA", "RMSSD", "SDNN"]

    num_ibi = len(ibi_values)
    num_windows = int(num_ibi/seg_size)

    segment_features = []
    seg_count = 0

    for n in range(0, num_windows):
        ibi_start_index = n*seg_size
        ibi_end_index = (n+1)*seg_size
        seg_count += 1

        ibi_subset = ibi_values[ibi_start_index:ibi_end_index]

        ibi_mean = np.mean(ibi_subset)
        hr_mean = (1 / (ibi_mean / 1000)) * 60

        result = td.sdnn(nni=ibi_subset)
        sdnn = result['sdnn']

        result = td.rmssd(nni=ibi_subset)
        rmssd = result['rmssd']

        result = fd.frequency_domain(nni=ibi_subset)
        rsa = result['fft_log'][2]

        segment_features.append([subj_name, task_name, seg_count, hr_mean, rsa, rmssd, sdnn])


        # hrv_tuple = pyhrv.hrv(nni=ibi_subset, plot_ecg=False, plot_tachogram=False, show=False)
        # plt.close('all')
        #
        # hrv_dict = dict(hrv_tuple)
        #
        # segment_features.append([subj_name, task_name, seg_count, hr_mean, hrv_dict['fft_abs'][1], hrv_dict['fft_abs'][2],
        #                  hrv_dict['fft_ratio'], hrv_dict['fft_norm'][0], hrv_dict['fft_norm'][1],
        #                  hrv_dict['fft_abs'][0], hrv_dict['fft_total'], hrv_dict['pnn50'],
        #                  hrv_dict['nn50'], hrv_dict['rmssd'], hrv_dict['sdnn'],
        #                  hrv_dict['tinn'], hrv_dict['sd1']])

        print('Segment Count: ' + str(seg_count))

    seg_df = pd.DataFrame(segment_features, columns=column_names)
    return seg_df