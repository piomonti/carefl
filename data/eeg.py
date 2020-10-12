"""
file to process and load EEG data
data can be downloaded from http://bbci.de/competition/iii/download/ after authentication
we used dataset 4a, subject 3 at 1MHz as in Lopez-paz et al 15, and Peters et al. 09
once downloaded to root (default = "data/eeg"), unzip to get this file: "data/eeg/1000mhz/data_set_IVa_av_cnt.txt"
"""
import numpy as np
import os


def create_timeseries(root='data/egg/'):
    try:
        print("loading {}".format(os.path.join(root, '1000Hz', 'data_set_IVa_av_cnt.txt')))
        eeg = np.loadtxt(os.path.join(root, '1000Hz', 'data_set_IVa_av_cnt.txt'))
    except OSError as e:
        print("Please download dataset 4a subject 3 at 1MHz from "
              "http://bbci.de/competition/iii/download/ after authentication and unzip it in '{}'".format(root))
        raise e
    eeg_5s = eeg[:5000]
    assert not np.any(np.isnan(eeg_5s))
    pieces = []
    for i in range(10):
        pieces.append(eeg_5s[i * 500:(i + 1) * 500])
    timeseries = np.hstack(pieces)
    directions = []
    shuffled_timeseries = np.zeros_like(timeseries)
    np.random.seed(0)
    for i in range(timeseries.shape[1]):
        if np.random.uniform() < .5:
            directions.append('y->x')
            shuffled_timeseries[:, i] = timeseries[::-1, i]
        else:
            directions.append('x->y')
            shuffled_timeseries[:, i] = timeseries[:, i]
    directions = np.array(directions).reshape((1, -1))
    print("saving to {}".format(os.path.join(root, 'eeg_1000Hz_5s_10pieces.txt')))
    np.savetxt(os.path.join(root, 'eeg_1000Hz_5s_10pieces.txt'), timeseries)
    np.savetxt(os.path.join(root, 'eeg_1000Hz_5s_10pieces_ts.txt'), shuffled_timeseries)
    np.savetxt(os.path.join(root, 'eeg_1000Hz_5s_10pieces_dirs.txt'), directions == 'x->y', fmt='%d')


# def eeg_data(root='data/eeg', idx=None, lag=None, n_obs=500):
#     if lag is None:
#         lag = [1]
#     try:
#         raw_data = np.loadtxt(os.path.join(root, 'eeg_1000Hz_5s_10pieces_ts.txt'), usecols=idx)
#     except OSError:
#         print("Raw data not processed - processing...")
#         create_timeseries(root)
#         raw_data = np.loadtxt(os.path.join(root, 'eeg_1000Hz_5s_10pieces_ts.txt'), usecols=idx)
#     if raw_data.ndim < 2:
#         raw_data = raw_data.reshape((-1, 1))
#     raw_data = raw_data[:n_obs]
#     data = []
#     for l in lag:
#         data.append(np.concatenate([raw_data[:-l], raw_data[l:]], axis=1))
#     data = np.concatenate(data, axis=0)
#     # load direction
#     direction = np.loadtxt(os.path.join(root, 'eeg_1000Hz_5s_10pieces_dirs.txt'), usecols=idx)
#     direction = 'x->y' if direction == 1 else 'y->x'
#     return data, direction


def eeg_data(root='data/eeg', idx=None, lag=None, n_obs=500):
    if lag is None:
        lag = [1]
    try:
        raw_data = np.loadtxt(os.path.join(root, 'eeg_1000Hz_5s_ts.txt'), usecols=idx)
    except OSError:
        print("Raw data not processed - processing...")
        create_timeseries(root)
        raw_data = np.loadtxt(os.path.join(root, 'eeg_1000Hz_5s_ts.txt'), usecols=idx)
    if raw_data.ndim < 2:
        raw_data = raw_data.reshape((-1, 1))
    # load direction
    direction = np.loadtxt(os.path.join(root, 'eeg_1000Hz_5s_dirs.txt'), usecols=idx)
    raw_data = raw_data[:n_obs] if direction == 1 else raw_data[-n_obs:]
    direction = 'x->y' if direction == 1 else 'y->x'
    data = []
    for l in lag:
        data.append(np.concatenate([raw_data[:-l], raw_data[l:]], axis=1))
    data = np.concatenate(data, axis=0)
    return data, direction