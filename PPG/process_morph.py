import numpy as np
import pandas as pd
import os, sys
from os.path import join as pjoin
from scipy.signal import resample
from scipy.stats import pearsonr
from scipy.integrate import simpson
import warnings

warnings.simplefilter('ignore')

from PPG.process_ibi import peak_det2
from PPG.process_ppg import filter_ppg


def acquire_morph_features(ppg, t_start=0, sampling_rate=450):
    features = []
    # Get Wave(s), segmented by their feet
    waves_feet, resampled_waves_feet = segment_waves_by_feet(ppg, t_start, sampling_rate)

    # Calculate template match correlation, and get indices of waves that pass template matching check
    corr_feet, quality_waves_idxs_feet, quality_waves_feet = template_match_feet(resampled_waves_feet)
    quality_waves = [waves_feet[x] for x in quality_waves_idxs_feet]

    # Get Fiducial Points for the Waves that pass Template Matching Quality
    fiducial_data = get_fiducials(quality_waves)

    # Using Fiducial Data, extract morphological features
    morph_data = extract_morph_features(fiducial_data)

    # Get mean feature(s) for the given window
    mean_feature_dict, feature_dict = convert_feature_means(morph_data)
    features.append(mean_feature_dict)

    # Return features as a dataframe
    feature_df = pd.DataFrame(features)

    # Add "ppg_" delineator to features
    feature_df = feature_df.add_prefix('ppg_')
    return feature_df


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
        # Select part of ppg wave corresponding to a given segment of interest
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
    quality_waves = []
    quality_waves_idxs = []
    correlations = []
    # Calculate correlation of the waves in a window with template wave
    correlations = []
    for i, resampled_wave in enumerate(resampled_waves):
        corr = pearsonr(resampled_wave, template_wave).statistic
        correlations.append(corr)
        # Get the waves and their indices that correspond to good quality waves after template matching
        if corr > threshold:
            quality_waves_idxs.append(i)
            quality_waves.append(resampled_wave)

    # Return the mean correlation across the ppg window, quality waves and their indices
    return np.mean(correlations), quality_waves_idxs, quality_waves


def extract_wave_fiducials(wave, tol=0.001):
    """ Function for extracting fiducial points from the original, first, second, and third derivative waveforms
    for downstream morphological feature extraction

    Possible fiducial points include:
    - Original wave: onset, systole, dicrotic, diastole, end
    - First derivative wave: ms
    - Second derivative wave: a-,b-,c-,d-,e-
    - Third derivative wave: p1, p2

    Args:
        tol: tolerance value introduced, since zero crossing point of first derivative may not correspond to max,
            but something very close to max, per the case of determining if the detected ms fiducial point
            is valid.
    Output:
        fiducial_dict: Dictionary of fiducial points detectable for the given waveform.
            - Bad Waveform:  Return empty dictionary.
            - Acceptable Waveform: Return dictionary with: onset, systole, end, ms
            - Good Waveform: Return dictionary with: onset, systole, dicrotic, diastole, end, ms

        wave_class: The classification of the determined by fiducial points, denominated by:
            - Class 1: Bad Waveform
            - Class 2: Acceptable Waveform
            - Class 3: Good Waveform
            - Class 4: Acceptable Waveform, but rejected due to inappropriate ms fiducial point
            - Class 5: Good Waveform, but rejected due to inappropriate ms fiducial point
    """
    # Take the 1st, 2nd, 3rd, and 4th derivative of the wave
    wave_first = np.vstack([wave[:, 0], np.gradient(wave[:, 1])]).T
    wave_second = np.vstack([wave[:, 0], np.gradient(wave_first[:, 1])]).T
    wave_third = np.vstack([wave[:, 0], np.gradient(wave_second[:, 1])]).T
    wave_fourth = np.vstack([wave[:, 0], np.gradient(wave_third[:, 1])]).T
    # Get the zero-crossing points for original, 1st - 3rd derivative, corresponding to fiducial points
    zero_crossings = np.where(np.diff(np.sign(wave_first[:, 1])))[0]
    zero_crossings_first = np.where(np.diff(np.sign(wave_second[:, 1])))[0]
    zero_crossings_second = np.where(np.diff(np.sign(wave_third[:, 1])))[0]
    zero_crossings_third = np.where(np.diff(np.sign(wave_fourth[:, 1])))[0]
    fiducial_dict = dict()
    # Wave Classes: 1 - bad, 2 - acceptable, 3- good 4- acceptable, bad , 5-good, bad

    if len(zero_crossings) == 5:  # Possibly Good Waveform
        # Original wave fiducial points
        fiducial_dict['onset'] = wave[zero_crossings[0]]
        fiducial_dict['systole'] = wave[zero_crossings[1]]
        fiducial_dict['dicrotic'] = wave[zero_crossings[2]]
        fiducial_dict['diastole'] = wave[zero_crossings[3]]
        fiducial_dict['end'] = wave[zero_crossings[4]]
        # First derivative fiducial points
        fiducial_dict['ms'] = wave_first[zero_crossings_first[0]]
        # Second derivative fiducial points
        fiducial_dict['a'] = wave_second[zero_crossings_second[0]]
        fiducial_dict['b'] = wave_second[zero_crossings_second[1]]
        fiducial_dict['c'] = wave_second[zero_crossings_second[2]]
        if len(zero_crossings_second) > 3:
            fiducial_dict['d'] = wave_second[zero_crossings_second[3]]
        if len(zero_crossings_second) > 4:
            fiducial_dict['e'] = wave_second[zero_crossings_second[4]]
        # Third derivative fiducial points - may need modification since signal result may not be precise
        # TODO : p1 and p2 are early and late systoles points, p1 is the first maxima after b- wave
        fiducial_dict['p1'] = wave_third[zero_crossings_third[2]]
        fiducial_dict['p2'] = wave_third[zero_crossings_third[3]]

        ''' Additional Check(s)'''
        # Check that systolic peak x-value greater than dicrotic notch & diastolic peak x-values
        if fiducial_dict['dicrotic'][1] > fiducial_dict['systole'][1] or fiducial_dict['diastole'][1] > \
                fiducial_dict['systole'][1]:
            return fiducial_dict, 5
        # Check that maximum upslope (time) comes before the systolic peak (time) ; check that ms is the highest value in first deriv
        elif (fiducial_dict['ms'][0] > fiducial_dict['systole'][0]) or (
                np.max(wave_first[:, 1]) > fiducial_dict['ms'][1] + tol):

            return fiducial_dict, 5
        else:
            return fiducial_dict, 3

    elif len(zero_crossings) == 3:  # Possibly Acceptable Waveform
        # Original wave fiducial points
        fiducial_dict['onset'] = wave[zero_crossings[0]]
        fiducial_dict['systole'] = wave[zero_crossings[1]]
        fiducial_dict['end'] = wave[zero_crossings[2]]
        # First derivative fiducial points
        fiducial_dict['ms'] = wave_first[zero_crossings_first[0]]
        # Second derivative fiducial points
        fiducial_dict['a'] = wave_second[zero_crossings_second[0]]
        if len(zero_crossings_second) > 1:
            fiducial_dict['b'] = wave_second[zero_crossings_second[1]]
        if len(zero_crossings_second) > 2:
            fiducial_dict['c'] = wave_second[zero_crossings_second[2]]
        if len(zero_crossings_second) > 3:
            fiducial_dict['d'] = wave_second[zero_crossings_second[3]]
        if len(zero_crossings_second) > 4:
            fiducial_dict['e'] = wave_second[zero_crossings_second[4]]
        # Third derivative fiducial points - may need modification since signal result may not be precise
        if len(zero_crossings_third) > 2:
            fiducial_dict['p1'] = wave_third[zero_crossings_third[2]]
        if len(zero_crossings_third) > 3:
            fiducial_dict['p2'] = wave_third[zero_crossings_third[3]]
        ''' Additional Check(s)'''
        # Check that maximum upslope (time) comes before the systolic peak (time) ; check that ms is the highest value in first deriv
        if (fiducial_dict['ms'][0] > fiducial_dict['systole'][0]) or (
                np.max(wave_first[:, 1]) > fiducial_dict['ms'][1] + tol):
            return fiducial_dict, 4
        else:
            return fiducial_dict, 2
    else:  # Bad Waveform
        return fiducial_dict, 1


def get_fiducials(waves):
    # Function to collect all fiducials for the waves of a given sample signal, return as a list of dictionaries
    fiducial_data = []
    for i, wave in enumerate(waves):
        fiducial_dict, class_val = extract_wave_fiducials(wave)
        fiducial_dict['wave_class'] = class_val
        fiducial_dict['wave_number'] = i + 1
        fiducial_dict['segmented_wave'] = wave
        fiducial_data.append(fiducial_dict)
    return fiducial_data


def extract_timing_features(fiducial_dict):
    """
    Acceptable Wave Features:
    - Pulse Time: t(end) - t(onset)
    - Crest Time: t(systole) - t(onset)
    - MS Time: t(ms) - t(onset)
    - IPR: 60/(Pulse Time)
    - t_b_c : t(c) - t(b)

    Excellent Wave Feature(s) (includes acceptable wave feature(s) too):
    - Delta_T : t(dia) - t(sys)
    t_sys : t(dic) - t(onset)
    t_dia : t(end) - t(dic)
    t_ratio : t_sys / t_dia
    t_p1_dia : t(dia) - t(p1)
    t_p2_dia : t(dia) - t(p2)
    t_b_d : t(d) - t(b)
    """

    feature_dict = dict()

    # Excellent Wave Features
    if fiducial_dict['wave_class'] == 3:
        feature_dict['Delta_T'] = fiducial_dict['diastole'][0] - fiducial_dict['systole'][0]
        feature_dict['t_sys'] = fiducial_dict['dicrotic'][0] - fiducial_dict['onset'][0]
        feature_dict['t_dia'] = fiducial_dict['end'][0] - fiducial_dict['dicrotic'][0]
        feature_dict['t_ratio'] = feature_dict['t_sys'] / feature_dict['t_dia']
        feature_dict['t_p1_dia'] = fiducial_dict['diastole'][0] - fiducial_dict['p1'][0]
        feature_dict['t_p2_dia'] = fiducial_dict['diastole'][0] - fiducial_dict['p2'][0]
        if 'b' in fiducial_dict.keys() and 'd' in fiducial_dict.keys():
            feature_dict['t_b_d'] = fiducial_dict['d'][0] - fiducial_dict['b'][0]

    # Acceptable Wave Features
    feature_dict['Pulse_Time'] = fiducial_dict['end'][0] - fiducial_dict['onset'][0]
    feature_dict['Crest_Time'] = fiducial_dict['systole'][0] - fiducial_dict['onset'][0]
    feature_dict['MS_Time'] = fiducial_dict['ms'][0] - fiducial_dict['onset'][0]
    feature_dict['IPR'] = 60 / feature_dict['Pulse_Time']
    if 'b' in fiducial_dict.keys() and 'c' in fiducial_dict.keys():
        feature_dict['t_b_c'] = fiducial_dict['c'][0] - fiducial_dict['b'][0]
    return feature_dict


def extract_amplitude_features(fiducial_dict):
    """
    Acceptable Wave Features:
    - Pulse_Wave_Amplitude : x(sys) - x(onset)
    - MS_Amplitude : x(ms) / x(sys) - x(onset)
    - b_a : x(b) / x(a)
    - c_a: x(c) / x(a)

    Excellent Wave Features:
    - AI: (x(p2) - x(p1)) / (x(sys) - x(onset))
    - RI: (x(dia) - x(onset))/(x(sys) - x(onset))
    - RI_p1: (x(dia) - x(onset))/(x(p1) - x(onset))
    - RI_p2: (x(dia) - x(onset))/(x(p2) - x(onset))
    - ratio_p2_p1: (x(p2) - x(onset))/(x(p1) - x(onset))
    - d_a : x(d) / x(a)
    - e_a : x(e) / x(a)
    - AGI : (x(b) - x(c) - x(d) - x(e))/x(a)
    """
    feature_dict = dict()

    if fiducial_dict['wave_class'] == 3:
        feature_dict['AI'] = (fiducial_dict['p2'][1] - fiducial_dict['p1'][1]) / (
                    fiducial_dict['systole'][1] - fiducial_dict['onset'][1])
        feature_dict['RI'] = (fiducial_dict['diastole'][1] - fiducial_dict['onset'][1]) / (
                    fiducial_dict['systole'][1] - fiducial_dict['onset'][1])
        feature_dict['RI_p1'] = (fiducial_dict['diastole'][1] - fiducial_dict['onset'][1]) / (
                    fiducial_dict['p1'][1] - fiducial_dict['onset'][1])
        feature_dict['RI_p2'] = (fiducial_dict['diastole'][1] - fiducial_dict['onset'][1]) / (
                    fiducial_dict['p2'][1] - fiducial_dict['onset'][1])
        feature_dict['ratio_p2_p1'] = (fiducial_dict['p2'][1] - fiducial_dict['onset'][1]) / (
                    fiducial_dict['p1'][1] - fiducial_dict['onset'][1])
        if 'd' in fiducial_dict.keys():
            feature_dict['d_a'] = fiducial_dict['d'][1] / fiducial_dict['a'][1]
        if 'e' in fiducial_dict.keys() and 'd' in fiducial_dict.keys():
            feature_dict['e_a'] = fiducial_dict['e'][1] / fiducial_dict['a'][1]
            feature_dict['AGI'] = (fiducial_dict['b'][1] - fiducial_dict['c'][1] - fiducial_dict['d'][1] -
                                   fiducial_dict['e'][1]) / fiducial_dict['a'][1]

    feature_dict['Pulse_Wave_Amplitude'] = fiducial_dict['systole'][1] - fiducial_dict['onset'][1]
    feature_dict['Max_Slope'] = fiducial_dict['ms'][1] / (fiducial_dict['systole'][1] - fiducial_dict['onset'][1])
    feature_dict['MS_Amplitude'] = fiducial_dict['ms'][1] - fiducial_dict['onset'][1]
    if 'b' in fiducial_dict.keys():
        feature_dict['b_a'] = fiducial_dict['b'][1] / fiducial_dict['a'][1]
    if 'c' in fiducial_dict.keys():
        feature_dict['c_a'] = fiducial_dict['c'][1] / fiducial_dict['a'][1]
    return feature_dict


def extract_area_slope_features(fiducial_dict):
    """
    - requires information about the wave itself, since need to calculate wave under the curve

    Acceptable Wave Features:
    - slope(b_c):
    Excellent Wave Features:
    - A1: area between t(dic) , t(onset)
    - A2: area between t(dic), t(end)
    - IPA: A2/A1
    - IPAD: IPA + x(d)/x(a)
    - slope(b_d):
    TODO: Slope feature(s) not completed yet
    """
    feature_dict = dict()

    if fiducial_dict['wave_class'] == 3:

        # feature_dict['Total_Area'] = simpson(fiducial_dict['segmented_wave'][:,1] - np.min(fiducial_dict['segmented_wave'][:,1]), fiducial_dict['segmented_wave'][:,0], axis = 0) # Add offset (take min of wave, subtract which equals to addition since negative value, to start calculation at y = 0)
        x_dic = fiducial_dict['dicrotic'][0]
        x_onset = fiducial_dict['onset'][0]
        x_end = fiducial_dict['end'][0]
        A1_wave = fiducial_dict['segmented_wave'][np.where(
            (fiducial_dict['segmented_wave'][:, 0] < x_dic) & (fiducial_dict['segmented_wave'][:, 0] > x_onset))]
        feature_dict['A1'] = simpson(A1_wave[:, 1] - np.min(fiducial_dict['segmented_wave'][:, 1]), A1_wave[:, 0])

        A2_wave = fiducial_dict['segmented_wave'][
            np.where((fiducial_dict['segmented_wave'][:, 0] < x_end) & (fiducial_dict['segmented_wave'][:, 0] > x_dic))]
        feature_dict['A2'] = simpson(A2_wave[:, 1] - np.min(fiducial_dict['segmented_wave'][:, 1]), A2_wave[:, 0])

        feature_dict['IPA'] = feature_dict['A2'] / feature_dict['A1']
        if 'd' in fiducial_dict.keys():
            feature_dict['IPAD'] = feature_dict['IPA'] + fiducial_dict['d'][1] / fiducial_dict['a'][1]

        # wave_fft = np.abs(fft(fiducial_dict['segmented_wave'][:,1]))
        # numerator = np.sum(wave_fft[2:len(wave_fft)//2] ** 2)
        # denominator = np.sum(wave_fft[1:len(wave_fft)//2] ** 2)
        # nha = numerator/denominator
        # feature_dict['NHA'] = nha
        # feature_dict['IHAR'] = (1 - nha) / feature_dict['IPA']
    return feature_dict


def extract_morph_features(fiducial_data):
    morph_feature_data = []
    for fiducial_dict in fiducial_data:
        # Feature Extraction for Acceptable Waves
        if fiducial_dict['wave_class'] == 2 or fiducial_dict['wave_class'] == 3:
            # Usable points: onset, systole, end, ms, a, b, c (d, e, p1, p2)
            timing_features = extract_timing_features(fiducial_dict)
            amplitude_features = extract_amplitude_features(fiducial_dict)
            area_features = extract_area_slope_features(fiducial_dict)
            feature_dict = {**timing_features, **amplitude_features, **area_features}
            feature_dict['wave_number'] = fiducial_dict['wave_number']
            feature_dict['wave_class'] = fiducial_dict['wave_class']
            morph_feature_data.append(feature_dict)
        else:
            continue
    return morph_feature_data


def convert_feature_means(feature_data):
    # For the features, concatenate all features to a given key list in a dictionary mapping features and individual wave values
    feature_dict = dict()
    for feat_dict in feature_data:
        feat_names = list(feat_dict.keys())
        for feat_name in feat_names:
            if feat_name not in list(feature_dict.keys()):
                feature_dict[feat_name] = [feat_dict[feat_name]]
            else:
                feature_dict[feat_name].append(feat_dict[feat_name])

    # Calculate mean feature(s) from dictionary map of features and individual wave values
    mean_feature_dict = dict()
    feat_names = list(feature_dict.keys())
    for feat_name in feat_names:
        if feat_name != 'wave_class' and feat_name != 'wave_number':
            mean_feature_dict[feat_name] = np.mean(feature_dict[feat_name])
    return mean_feature_dict, feature_dict