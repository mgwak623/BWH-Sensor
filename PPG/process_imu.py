from scipy import signal
import numpy as np

SAMPLING_RATE = 25

# Filter Definitions
NZEROS_BACK = 250
NZEROS_BACK_PLUS_ONE = NZEROS_BACK # Alternate: NZEROS_BACK + 1
NZEROS_LOC = 25
NZEROS_LOC_PLUS_ONE = NZEROS_LOC # Alternate: NZEROS_LOC + 1

# Bandpass Coefficients for ACC Filter
bcoeffs = [-0.0102402099986476, -0.00990257793931302, -0.00750747831363377, -0.00422497631469390, -0.00191531794429593,
         -0.00183045716885961, -0.00362397271834792, -0.00543851713191903, -0.00512335769949442, -0.00180389860314510,
         0.00331730391690034, 0.00760026334554545, 0.00873110441766462, 0.00643449542790326, 0.00287547173134653,
         0.00131950416988938, 0.00384718660086280, 0.00969138920426109, 0.0154731238969883, 0.0173636297567946,
         0.0138047228100204, 0.00687317534436962, 0.00115763628330500, 0.000696715335756891, 0.00598414553146245,
         0.0132107265593129, 0.0165338424780570, 0.0120738746894926, 0.000955130323087103, -0.0109028332399347,
         -0.0164696106332548, -0.0125330566974359, -0.00249519762892161, 0.00522616849888151, 0.00260706487307444,
         -0.0121144190245031, -0.0322145578529596, -0.0459858042996077, -0.0443237413488410, -0.0276953263813468,
         -0.00754347961021164, -0.000259940262277672, -0.0165840271069756, -0.0527260288929336, -0.0892830441379052,
         -0.0998716029915171, -0.0655844669337435, 0.0126276531903131, 0.111125046569517, 0.192853173045798,
         0.224527044031722, 0.192853173045798, 0.111125046569517, 0.0126276531903131, -0.0655844669337435,
         -0.0998716029915171, -0.0892830441379052, -0.0527260288929336, -0.0165840271069756, -0.000259940262277672,
         -0.00754347961021164, -0.0276953263813468, -0.0443237413488410, -0.0459858042996077, -0.0322145578529596,
         -0.0121144190245031, 0.00260706487307444, 0.00522616849888151, -0.00249519762892161, -0.0125330566974359,
         -0.0164696106332548, -0.0109028332399347, 0.000955130323087103, 0.0120738746894926, 0.0165338424780570,
         0.0132107265593129, 0.00598414553146245, 0.000696715335756891, 0.00115763628330500, 0.00687317534436962,
         0.0138047228100204, 0.0173636297567946, 0.0154731238969883, 0.00969138920426109, 0.00384718660086280,
         0.00131950416988938, 0.00287547173134653, 0.00643449542790326, 0.00873110441766462, 0.00760026334554545,
         0.00331730391690034, -0.00180389860314510, -0.00512335769949442, -0.00543851713191903, -0.00362397271834792,
         -0.00183045716885961, -0.00191531794429593, -0.00422497631469390, -0.00750747831363377, -0.00990257793931302,
         -0.0102402099986476]

acoeffs = [1]

# Coefficients for highpass Filter
acoeffHigh = [1, -0.881618592363189]
bcoeffHigh = [0.940809296181595, -0.940809296181595]

# Coefficients for motion detection filters
xcoeffs_back = [0.004] * NZEROS_BACK_PLUS_ONE
xcoeffs_loc = [0.04] * NZEROS_LOC_PLUS_ONE


def bandpass_filter(acc_X_input, acc_Y_input, acc_Z_input, order=4):

    imu_nyq = 0.5 * SAMPLING_RATE

    b, a = signal.butter(order, [0.5 / imu_nyq, 10 / imu_nyq], btype='bandpass')
    acc_filter_X = signal.filtfilt(b, a, acc_X_input, method='gust')
    acc_filter_Y = signal.filtfilt(b, a, acc_Y_input, method='gust')
    acc_filter_Z = signal.filtfilt(b, a, acc_Z_input, method='gust')

    return acc_filter_X, acc_filter_Y, acc_filter_Z


def motion_detection(acc_x_raw, acc_y_raw, acc_z_raw):

    db_threshold = 1

    acc_filter_x = signal.filtfilt(bcoeffs, acoeffs, acc_x_raw)
    acc_filter_y = signal.filtfilt(bcoeffs, acoeffs, acc_y_raw)
    acc_filter_z = signal.filtfilt(bcoeffs, acoeffs, acc_z_raw)

    axl_power = []

    for i in range(0, len(acc_filter_x)):
        axl_power.append(np.sqrt(np.square(acc_filter_x[i]) + np.square(acc_filter_y[i]) + np.square(acc_filter_z[i])))

    y_back = signal.filtfilt(xcoeffs_back, acoeffs, axl_power)
    y_loc = signal.filtfilt(xcoeffs_loc, acoeffs, axl_power)

    motion_flags = []

    for i in range(0, len(y_back)):
        if (y_loc[i]/y_back[i]) >= np.power(10, db_threshold/10):
            motion_flags.append(1)
        else:
            motion_flags.append(0)

    return motion_flags


