"""
file to process and load EEG data
data can be downloaded from http://bbci.de/competition/iii/download/ after authentication
we used dataset 4a, subject 3 at 1MHz
once downloaded to data root (default = "data/eeg"), unzip to get this file: "data/eeg/1000mhz/data_set_IVa_av_cnt.txt"
"""
import numpy as np
import os


def create_timeseries(root='data/egg/'):
    """
    Prepreocesses the EEG dataset, flipping the time arrow of certain channels at random.
    The preprocessed data is saved to 3 different files:
        - `eeg_1000Hz_5s` contains the dataset before shuffling
        - `eeg_1000Hz_5s_ts` contains the dataset, after certain channels have been flipped in time.
        - `eeg_1000Hz_5s_dirs` contains the direction of each channel (1 is forward, 0 is backward).
    """
    try:
        print("loading {}".format(os.path.join(root, '1000Hz', 'data_set_IVa_av_cnt.txt')))
        eeg = np.loadtxt(os.path.join(root, '1000Hz', 'data_set_IVa_av_cnt.txt'))
    except OSError as e:
        print("Please download dataset 4a subject 3 at 1MHz from "
              "http://bbci.de/competition/iii/download/ after authentication and unzip it in '{}'".format(root))
        raise e
    eeg_5s = eeg[:5000]
    assert not np.any(np.isnan(eeg_5s))
    directions5s = []
    shuffled_timeseries5s = np.zeros_like(eeg_5s)
    np.random.seed(0)
    for i in range(eeg_5s.shape[1]):
        if np.random.uniform() < .5:
            directions5s.append('y->x')
            shuffled_timeseries5s[:, i] = eeg_5s[::-1, i]
        else:
            directions5s.append('x->y')
            shuffled_timeseries5s[:, i] = eeg_5s[:, i]
    directions5s = np.array(directions5s).reshape((1, -1))
    np.savetxt(os.path.join('data/eeg', 'eeg_1000Hz_5s.txt'), eeg_5s)
    np.savetxt(os.path.join('data/eeg', 'eeg_1000Hz_5s_ts.txt'), shuffled_timeseries5s)
    np.savetxt(os.path.join('data/eeg', 'eeg_1000Hz_5s_dirs.txt'), directions5s == 'x->y', fmt='%d')



def eeg_data(root='data/eeg', idx=None, lag=None, n_obs=500):
    """
    Reads EEG channels from preprocessed data.

    The EEG data has n channels. This function reads the `idx`-th channel as $x$,
    delays it by `lag` and stores it as $y$, and returns the concatenated 2-d array
    $(x, y)$ of shape `(n_obs-1, 2)`.

    The task is then to guess to guess the time arrow, by deciding if x->y or y->x.

    Parameters:
    ----------
    idx: int or array_like
      Index or list of indices of the channels to load
      Note that if `idx` is an array_like, the returned array will be of size
      (..., len(idx)*2) instead of (..., 2).
    lag: int or array_like
      Lag or lags used to compute $y$. If 2 or more lags are given, create
      a $y$ for each lag value, and concatenate the results along the 0-th axis
    n_obs: int
      Number of observations to use from the preprocessed EEG data.
      Note that if multiple lag values are provided, the returned array will not be
      of size `(n_obs-1, 2)`, but rather `(len(lag)*n_obs - sum_{i=1}^len(lag)i, 2)`

    Returns:
    ----------
    data: ndarray, shape (..., len(idx)*2)
        Array containting the original and delayed channels
    direction: ndarray, shape (..., len(idx))
        Array containing the direction of each channel.
        The values in this array can be either 'x->y' or 'y->x'
    """
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
