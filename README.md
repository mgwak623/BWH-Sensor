# BWH-sensor
Python script to generate PPG features from BWH-TAVR study.

## Prerequisites

Installing dependencies
```commandline
pip install poetry
poetry install
```

## Usage

```commandline
python3 main.py <ppg_file_path> <sampling_rate> <segment_duration>
```
- ppg_file_path: raw PPG file path in csv
- sampling_rate: PPG sensor sampling rate. I believe it was 25 Hz before but now 100 Hz.
- segment_duration: non-overlapping window size. i.e., 1min or 5min

Example:
```commandline
python3 main.py Data/ppg/BWHStudy2-124-1732075200000-1732078800000-PpgGreen.csv 25 5min
```
