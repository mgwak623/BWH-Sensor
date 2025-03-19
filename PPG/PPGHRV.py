import pandas as pd
from PPG.process_ppg import filter_ppg
import numpy as np
from PPG.process_ibi import peak_det2
from PPG.hrv_feature_extraction import get_hrv_features
from PPG.process_imu import motion_detection
from PPG.process_ppg import get_hr_fft
from PPG.process_ppg import fine_filter_ppg
import warnings

warnings.simplefilter('ignore')
from scipy.signal import resample
from scipy.stats import pearsonr
from scipy.integrate import simpson
from PPG.process_morph import acquire_morph_features

''' Methods for template matching '''


def segment_waves_by_feet(ppg, t_start=0, sample_rate=25):
    # Invert signal and perform peak detection to get feet
    ppg_inverted = -ppg
    IBI_threshold = (0.3, 2)
    ppg_sim_ts = np.linspace(t_start, (t_start) + len(ppg_inverted) / sample_rate, len(ppg_inverted))
    peaks, outlier_list, peak_list, ibi, ibi_ts = peak_det2(ppg_inverted, ppg_sim_ts, ibi_threshold=IBI_threshold,
                                                            fs=450)
    feet = np.vstack([peaks[:, 0], -peaks[:, 1]]).T
    ppg_timestamped = np.vstack([ppg_sim_ts, ppg]).T

    # Segment waves based on feet. Resample to equal lengths in order to construct template
    # and perform comparison
    waves = []
    resampled_waves = []
    for i in range(len(feet) - 1):
        # Select part of segment corresponding to a given wave
        offset_start = 0.05  # Offsets present originally for morphological feature extraction
        offset_end = 0.05  # However, this should not affect the template construction in HRV extraction
        wave_start, wave_end = feet[i][0] - offset_start, feet[i + 1][0] + offset_end
        wave_idxs = np.where(np.logical_and(ppg_timestamped[:, 0] > wave_start, ppg_timestamped[:, 0] <= wave_end))
        wave = ppg_timestamped[wave_idxs, :].squeeze()
        # Resample wave for template matching
        resampled_wave = resample(wave[:, 1], 256)
        resampled_waves.append(resampled_wave)
        waves.append(wave)
    return waves, resampled_waves


def template_match_feet(resampled_waves, threshold=0.9):
    """
    Using resampled waves, calculate average correlation.
    """
    # Determine which wave(s) are quality waves via template matching
    template_wave = np.average(np.array(resampled_waves), axis=0)

    # Calculate correlation of the waves in a window with template wave
    correlations = []
    for i, resampled_wave in enumerate(resampled_waves):
        corr = pearsonr(resampled_wave, template_wave).statistic
        correlations.append(corr)

    # Return the mean correlation across the ppg window
    return np.mean(correlations)


def check_quality(ppg, sampling_rate=25, threshold=0.9):
    """
    Check the quality of the given segment
    """
    # Segment waves
    t_start = 0
    waves_feet, resampled_waves_feet = segment_waves_by_feet(ppg, t_start, sampling_rate)

    # Calculate mean correlation to template
    corr = template_match_feet(resampled_waves_feet)

    # Return true if mean correlation > threshold (no fine filter needed) ; else false (fine filter needed)
    if corr > threshold:
        return False
    else:
        return True


''' PPG Class '''


class PPG():
    def __int__(self):
        self.window_df = None

    def reactivity(self, original_df, baseline_df):
        for column in original_df.columns:
            if column in original_df.columns and column in baseline_df.columns:
                if "Pulse_Wave_Amp_Mean" == column:
                    original_df[column + "_reactivity"] = ((original_df[column] - baseline_df[column]) / baseline_df[
                        column]) * 100
                else:
                    original_df[column + "_reactivity"] = original_df[column] - baseline_df[column]
        return original_df

    # Start implementing from here. Return a dataframe with all features
    def generate(self, window_df, imu_df, window=0.5, sampling_rate=25, IBI_threshold=(0.3, 2), band=[0.7, 3.5]):
        self.window_df = window_df
        window_df = window_df.rename(columns={'ts': 'kernelTs', 'ppg': 'v.ppg.ppg5List'})

        # if sampling_rate == 450:
        #     # Change the timestamp and ppg signal name to kernelTS & 'v.ppg.ppg5List', invert signal
        #     window_df = window_df.rename(columns={'ts': 'kernelTs', 'ppg': 'v.ppg.ppg5List'})
        #
        # if sampling_rate == 1000:
        #     # Change the timestamp and ppg signal name to kernelTS & 'v.ppg.ppg5List'
        #     window_df = window_df.rename(columns={'ts': 'kernelTs', 'PPG': 'v.ppg.ppg5List'})

        # taskTimes = input_df['phoneTs']
        taskTimes = window_df['kernelTs']
        taskStart = taskTimes.iloc[0]
        taskEnd = taskTimes.iloc[-1]

        # In case kernelTs was reset
        if taskEnd < taskStart:
            input_len = len(window_df['kernelTs'])
            taskEnd = taskStart + (40 * (input_len - 1))

        ppg_all = window_df['v.ppg.ppg5List']

        percent_flag_motion = 0
        # Only necessary if you have IMU data
        if len(imu_df) > 0:

            acc_x = imu_df['v.accX']
            acc_y = imu_df['v.accY']
            acc_z = imu_df['v.accZ']

            min_window = window  # Minimum minutes of data to be processed

            IMU_SAMPLING_RATE = 25

            min_sample_acc = int(IMU_SAMPLING_RATE * 60 * min_window)

            # Motion artifact detection
            if len(acc_x) >= min_sample_acc:
                motion_flags = motion_detection(acc_x, acc_y, acc_z)

                count_flag_segment = np.sum(motion_flags)
                percent_flag_motion = (count_flag_segment / len(motion_flags)) * 100

        min_window = window  # Minimum minutes of data to be processed

        PPG_SAMPLING_RATE = sampling_rate

        min_sample_ppg = int(PPG_SAMPLING_RATE * 60 * min_window)

        ppg_len = len(ppg_all)

        fine = False  # Flag to determine if dynamic fine filtering should be used

        # Specific case for WBD POC Buds which need signal to be inverted
        ppg_filtered = filter_ppg(ppg_all, sampling_rate=sampling_rate, band=band)

        if sampling_rate == 450:  # Inversion done for POC data, as POC/WBD sensor detects "absence of blood volume" - Contact Li Zhu if any questions
            ppg_filtered = -ppg_filtered

        # Determine if dynamic fine filtering is necessary
        fine = check_quality(ppg_filtered, sampling_rate=sampling_rate)
        if fine == True:
            est_hr = get_hr_fft(ppg_filtered, 40, 180, sampling_rate=sampling_rate)
            ppg_narrow_filtered = fine_filter_ppg(ppg_filtered, est_hr, sampling_rate=sampling_rate)
            ppg_va = ppg_narrow_filtered
        else:
            ppg_va = ppg_filtered

        # Extract IBIs
        ppg_sim_ts = np.linspace(taskStart / 1000, (taskStart / 1000) + len(ppg_va) / PPG_SAMPLING_RATE,
                                 len(ppg_va))
        IBI_threshold = IBI_threshold
        _, outlier_list, peak_list, ibi, ibi_ts = peak_det2(ppg_va, ppg_sim_ts,
                                                            ibi_threshold=IBI_threshold, fs=sampling_rate)

        outlier_count = len(outlier_list)
        valid_ibi_count = len(ibi)

        percent_outlier = (outlier_count / (valid_ibi_count + outlier_count)) * 100

        valid_ibi_values = ibi * 1000
        valid_ibi_times = ibi_ts * 1000

        # Get Feature(s)
        hrv_df = get_hrv_features(valid_ibi_values)

        # Get Morphological Features
        morph_df = acquire_morph_features(ppg_filtered, taskStart, sampling_rate=sampling_rate)

        # Join Data
        ppg_feature_df = hrv_df.join(morph_df)

        ppg_feature_df['ppg_Outlier_Percent'] = percent_outlier

        # If IMU data exists, get motion percentage
        if len(imu_df) > 0:
            ppg_feature_df['ppg_Motion_Percent'] = percent_flag_motion

        return ppg_feature_df