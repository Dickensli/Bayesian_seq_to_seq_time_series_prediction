import h5py
import time
import pickle as pkl
import pandas as pd
import numpy as np
import os.path
import os
import argparse
import logging

import extractor
from feeder import VarFeeder
import numba
from typing import Tuple, Dict, Collection, List

log = logging.getLogger('makeFeatures')

class IdMap():
    def __init__(self):
        self.id2name_dict = dict()
        self.name2id_dict = dict()

    def read_pickle(self, load_path):
        """
        load_path : path to load the pickle
        """
        with open(load_path, 'rb') as handle:
            self.id2name_dict, self.name2id_dict = pkl.load(handle)

    def write_pickle(self, save_path, names):
        """
        names : pandas index series
        save_path : path to save the pickle
        """
        for i, name in enumerate(names):
            self.name2id_dict[name] = i
            self.id2name_dict[i] = name
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(save_path, 'wb') as handle:
            pkl.dump([self.id2name_dict, self.name2id_dict], handle, protocol=pkl.HIGHEST_PROTOCOL)

    def name2id(self, names):
        """
        names : a sequence of / a hash names for vms
        """
        assert len(self.name2id_dict.keys()) != 0
        if isinstance(names, list):
            return [self.name2id_dict.get(name, 0) for name in names]
        if isinstance(names, str):
            return self.name2id_dict.get(names, 0)

    def id2name(self, ids):
        """
        ids : a sequence of / an integers each representing a unique vm
        """
        assert len(self.id2name_dict.keys()) != 0
        if isinstance(ids, list):
            return [self.id2name_dict.get(id, 0) for id in ids]
        if isinstance(ids, str):
            return self.id2name_dict.get(ids, 0)


def fetch_target_data(dfs, name, raw_data, log_trans=True):
    """preprocess the raw data.
    dfs : list of pandas DataFrames
        df_cpu_max, df_cpu_num, df_mem
    name : string
        vm name
    raw_data : np array
        raw data get from Dataset
    log_trans : bool
        whether do log transform

    Returns:
    numerical_feas : list of pandas dataframes
        [cpu_max, vm_cpu_num, vm_mem_size]
    """
    features = [1, raw_data.shape[1] - 5, raw_data.shape[1] - 4]
    for i in range(len(dfs)):
        numerical_feas = raw_data[:, features[i]]
        if log_trans and features[i] == 1:
            numerical_feas = np.log1p(numerical_feas)
        dfs[i] = dfs[i].append(pd.Series(name=name, data=numerical_feas))

@numba.jit(nopython=True)
def fill_nan(data):
    """
    data : np array
    fill missing or 0 entry with previous observed data
    """
    for i in range(1,data.shape[0]):
        if data[i,1]==0 and data[i-1,1]!=0:
            if i<288:
                data[i,1] = np.mean(data[:i,1])
            else:
                data[i,1] = data[i-288,1]
            data[i,2:]=data[i-1,2:]
    return data

def read_hdf5(data_path="") -> List[pd.DataFrame]:
    """
    read all the hdf5 files in the data_path
    :param data_path: str the direction stors hdf5s
    :return: dict  {vim file name: dataframe....}
    """
    dataset = dict()
    files = os.listdir(data_path)
    if len(files)==0:
        raise ValueError("No hdf5 file in %s" % data_path)
    dfs = [pd.DataFrame() for idx in range(3)]
    for index, vim_file in enumerate(files):
        #print index
        vim_path = os.path.join("%s/%s" % (data_path, vim_file))
        data = h5py.File(vim_path, 'r')
        fetch_target_data(dfs, vim_file, fill_nan(np.array(data['data'])), log_trans=True)
    return dfs

def read_all(ori_data_path) -> List[pd.DataFrame]:
    data_path = os.path.realpath(ori_data_path)
    #return read_hdf5(data_path=data_path)
    return [pd.read_hdf(os.path.join(ori_data_path, "cpu_max.hdf5")), 
            pd.read_hdf(os.path.join(ori_data_path, "vm_cpu_num.hdf5")),
            pd.read_hdf(os.path.join(ori_data_path, "vm_mem_size.hdf5"))]

def read_x(ori_data_path, start, end) -> List[pd.DataFrame]:
    """
    Gets source data from start to end date. Any date can be None
    """
    dfs = read_all(ori_data_path)
    for i in range(len(dfs)):
        dfs[i] = dfs[i].sort_index()
    if start and end:
        return [df.iloc[:, start:end] for df in dfs]
    elif end:
        return [df.iloc[:, :end] for df in dfs]
    else:
        return dfs

@numba.jit(nopython=True)
def single_autocorr(series, lag):
    """
    Autocorrelation for single data series
    :param series: usage series
    :param lag: lag, days
    :return:
    """
    s1 = series[lag:]
    s2 = series[:-lag]
    if s1.shape[0] == 0 or s2.shape[0] == 0:
        return 0
    ms1 = np.mean(s1)
    ms2 = np.mean(s2)
    ds1 = s1 - ms1
    ds2 = s2 - ms2
    divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
    return np.sum(ds1 * ds2) / divider if divider != 0 else 0


@numba.jit(nopython=True)
def batch_autocorr(data, lag, starts, ends, threshold, backoffset=0):
    """
    Calculate autocorrelation for batch (many time series at once)
    :param data: Time series, shape [n_vm, n_time]
    :param lag: Autocorrelation lag
    :param starts: Start index for each series
    :param ends: End index for each series
    :param threshold: Minimum support (ratio of time series length to lag) to calculate meaningful autocorrelation.
    :param backoffset: Offset from the series end, timestamps.
    :return: autocorrelation, shape [n_series]. If series is too short (support less than threshold),
    autocorrelation value is NaN
    """
    n_series = data.shape[0]
    n_days = data.shape[1]
    max_end = n_days - backoffset
    corr = np.empty(n_series, dtype=np.float64)
    support = np.empty(n_series, dtype=np.float64)
    for i in range(n_series):
        series = data[i]
        end = min(ends[i], max_end)
        real_len = end - starts[i]
        support[i] = real_len/lag
        if support[i] > threshold:
            series = series[starts[i]:end]
            c_minus1 = single_autocorr(series, lag)
            c = single_autocorr(series, lag-1)
            c_plus1 = single_autocorr(series, lag+1)
            # Average value between exact lag and two nearest neighborhs for smoothness
            corr[i] = 0.5 * c_minus1 + 0.25 * c + 0.25 * c_plus1
        else:
            corr[i] = np.NaN
    return corr


@numba.jit(nopython=True)
def find_start_end(data: np.ndarray):
    """
    Calculates start and end of real usage data. Start is an index of first non-zero, non-NaN value,
     end is index of last non-zero, non-NaN value
    :param data: Time series, shape [n_vm, n_time]
    :return:
    """
    n_vm = data.shape[0]
    n_days = data.shape[1]
    start_idx = np.full(n_vm, -1, dtype=np.int32)
    end_idx = np.full(n_vm, -1, dtype=np.int32)
    for vm in range(n_vm):
        # scan from start to the end
        for day in range(n_days):
            if not np.isnan(data[vm, day]) and data[vm, day] > 0:
                start_idx[vm] = day
                break
        # reverse scan, from end to start
        for day in range(n_days - 1, -1, -1):
            if not np.isnan(data[vm, day]) and data[vm, day] > 0:
                end_idx[vm] = day
                break
    return start_idx, end_idx


def prepare_data(ori_data_path, start, end, valid_threshold) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[pd.DataFrame]]:
    """
    Reads source data, calculates start and end of each series, drops bad series, calculates log1p(series)
    :param start: start date of effective time interval, can be None to start from beginning
    :param end: end date of effective time interval, can be None to return all data
    :param valid_threshold: minimal ratio of series real length to entire (end-start) interval. Series dropped if
    ratio is less than threshold
    :return: tuple(log1p(series), nans, series start, series end)
    """
    dfs = read_x(ori_data_path, start, end)
    df = dfs[0]
    starts, ends = find_start_end(df.values)
    # boolean mask for bad (too short) series
    vm_mask = (ends - starts) / df.shape[1] < valid_threshold
    print("Masked %d vms from %d" % (vm_mask.sum(), len(df)))
    inv_mask = ~vm_mask
    df = df[inv_mask]
    for i in range(1, len(dfs)):
        dfs[i] = dfs[i][inv_mask]
    return df, starts[inv_mask], ends[inv_mask], dfs[1:]

def lag_indexes(begin, end) -> List[pd.Series]:
    """
    Calculates indexes for 1, 7 days backward lag for the given time range
    :param begin: start of time range
    :param end: end of time range
    :return: List of 2 Series, one for each lag. For each Series, index is time in range(begin, end), value is an index
     of target (lagged) time in a same Series. If target date is out of (begin,end) range, index is -1
    """
    index = np.arange(begin, end + 1)
    def lag(offset):
        offset *= 288
        lag_idx = index - offset
        return pd.Series(data=lag_idx.astype(np.int16)).apply(lambda x: -1 if x < 0 else x)

    return [lag(offset) for offset in (1, 7)]

def normalize(values: np.ndarray):
    return (values - values.mean()) / np.std(values)

def run(train_data_path="/nfs/isolation_project/intern/project/lihaocheng/vm", datadir='data',
        valid_threshold=0.04, predict_window=288, seasonal=1, corr_backoffset=0, **args):

    start_time = time.time()
    # Get the data
    df, starts, ends, dfs = prepare_data(train_data_path, args['start'], args['end'], valid_threshold)
    df_cpu_num = dfs[0]
    df_cpu_num = pd.concat([df_cpu_num] + [df_cpu_num.iloc[:, -1] for i in range(predict_window)], ignore_index=True, axis=1)

    log.debug("complete generating df_cpu_max and df_cpu_num, time elapse = %S", time.time() - start_time)
    # Our working date range
    data_start, data_end = df.columns[0], df.columns[-1]

    # We have to project some date-dependent features (day of week, etc) to the future dates for prediction
    features_end = data_end + predict_window
    print(f"start: {data_start}, end:{data_end}, features_end:{features_end}")
    features_time = features_end - data_start

    assert df.index.is_monotonic_increasing
    assert df_cpu_num.index.is_monotonic_increasing

    # daily autocorrelation
    day_autocorr = batch_autocorr(df.values, 288, starts, ends, 1.5, corr_backoffset)

    # weekly autocorrelation
    week_autocorr = batch_autocorr(df.values, 288 * 7, starts, ends, 2, corr_backoffset)

    # Normalise all the things
    day_autocorr = normalize(np.nan_to_num(day_autocorr))
    week_autocorr = normalize(np.nan_to_num(week_autocorr))

    # Make time-dependent features
    feature_time = np.arange(data_start, features_end + 1) % 288
    time_period = 288 / (2 * np.pi)
    dow_norm = feature_time / time_period
    dow = np.stack([np.cos(dow_norm), np.sin(dow_norm)], axis=-1)
    if seasonal > 1:
        for k in range(2, seasonal + 1):
            time_period = 288 / (2 * np.pi * k)
            dow_norm = feature_time / time_period
            dow = np.concatenate([dow, np.cos(dow_norm).reshape(-1,1), np.sin(dow_norm).reshape(-1,1)], axis=-1)

    # Assemble indices for quarterly lagged data
    lagged_ix = np.stack(lag_indexes(data_start, features_end), axis=-1)

    # Map vm names to integers and store them into a pickle
    toId = IdMap()
    toId.write_pickle(os.path.join(datadir, "toId.pkl"), df.index.values)

    # Assemble final output
    tensors = dict(
        usage=df,
        cpu_num=df_cpu_num,
        lagged_ix=lagged_ix,
        vm_ix=np.arange(len(df.index.values)),
        day_autocorr=day_autocorr,
        week_autocorr=week_autocorr,
        starts = starts,
        ends = ends,
        dow=dow,
    )
    plain = dict(
        features_time=features_time,
        data_time=len(df.columns),
        n_vm=len(df),
        data_start=data_start,
        data_end=data_end,
        features_end=features_end
    )

    # Store data to the disk
    VarFeeder(os.path.join(datadir, 'vars'), tensors, plain)
