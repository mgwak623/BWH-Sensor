import glob
import os
import sys
import pandas as pd
from PPG.PPGHRV import PPG

output_dir = 'Output/'

def segment_ppg(ppg_df, segment_dur='1min'):
    ppg_df['ts'] = ppg_df['timestamp'] + ppg_df['timeOffset']
    ppg_df['dt'] = pd.to_datetime(ppg_df['ts'], unit='ms')
    ppg_df = ppg_df.set_index("dt").sort_index()

    grouped = ppg_df.resample(segment_dur,
                              closed="right",
                              label="right")
    segments = [group for _, group in grouped]
    segments = [segment for segment in segments if not segment.empty]
    return segments

def save_to_csv(df, filename):
    os.makedirs(output_dir, exist_ok=True)   # Check if output directory exist

    filepath = f'{output_dir}/{filename}.csv'
    df.to_csv(filepath, index=False)

def generate_features(ppg_filepath, segment_duration, sampling_rate):
    features = PPG()
    ppg_df = pd.read_csv(ppg_filepath, sep='|', skiprows=1)
    imu_df = pd.DataFrame()

    segments = segment_ppg(ppg_df, segment_duration)
    ppg_features = list()
    for window_df in segments:
        feats_df = features.generate(window_df, imu_df, sampling_rate=sampling_rate)
        ppg_features.append(feats_df)

    ppg_features = pd.concat(ppg_features, ignore_index=True)
    print(ppg_features)
    filename = "features_" + os.path.basename(ppg_filepath)
    save_to_csv(ppg_features, filename)


if __name__ == "__main__":
    ppg_filename = sys.argv[1]
    sampling_rate = sys.argv[2]  # 25 Hz default
    segment_duration = sys.argv[3]   # "5min"

    generate_features(ppg_filename, segment_duration, int(sampling_rate))