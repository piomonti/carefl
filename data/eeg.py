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
    print("saving to {}".format(os.path.join(root, 'eeg_1000Hz_5s_10pieces.txt')))
    np.savetxt(os.path.join(root, 'eeg_1000Hz_5s_10pieces.txt'), timeseries)


def eeg_data(root='data/eeg', idx=None, lag=None, shuffle=False):
    if lag is None:
        lag = [1]
    try:
        raw_data = np.loadtxt(os.path.join(root, 'eeg_1000Hz_5s_10pieces.txt'), usecols=idx)
    except OSError:
        print("Raw data not processed - processing...")
        create_timeseries(root)
        raw_data = np.loadtxt(os.path.join(root, 'eeg_1000Hz_5s_10pieces.txt'), usecols=idx)
    if raw_data.ndim < 2:
        raw_data = raw_data.reshape((-1, 1))
    data = []
    for l in lag:
        data.append(np.concatenate([raw_data[:-l], raw_data[l:]], axis=1))
    data = np.concatenate(data, axis=0)
    dir = 'x->y'
    if shuffle:
        if np.random.uniform() < .5:
            d = raw_data.shape[1]
            data = np.concatenate([data[:, d:], data[:, :d]], axis=1)
            dir = 'y->x'
    return data, dir
