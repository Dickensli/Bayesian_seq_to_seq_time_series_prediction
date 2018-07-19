import pandas as pd
import numpy as np
import os.path
import os
import argparse

import extractor
from feeder import VarFeeder
import numba
from typing import Tuple, Dict, Collection, List
    
def read_all(ori_data_path) -> pd.DataFrame:
    data_path = os.path.realpath(ori_data_path)
    return pd.read_hdf(data_path).iloc[:, :, 0].T

def read_x(ori_data_path, start, end) -> pd.DataFrame:
    """
    Gets source data from start to end date. Any date can be None
    """
    df = read_all(ori_data_path)
    if start and end:
        return df.iloc[:, start:end]
    elif end:
        return df.iloc[:, :end]
    else:
        return df

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
    n_pages = data.shape[0]
    n_days = data.shape[1]
    start_idx = np.full(n_pages, -1, dtype=np.int32)
    end_idx = np.full(n_pages, -1, dtype=np.int32)
    for page in range(n_pages):
        # scan from start to the end
        for day in range(n_days):
            if not np.isnan(data[page, day]) and data[page, day] > 0:
                start_idx[page] = day
                break
        # reverse scan, from end to start
        for day in range(n_days - 1, -1, -1):
            if not np.isnan(data[page, day]) and data[page, day] > 0:
                end_idx[page] = day
                break
    return start_idx, end_idx


def prepare_data(ori_data_path, start, end, valid_threshold) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Reads source data, calculates start and end of each series, drops bad series, calculates log1p(series)
    :param start: start date of effective time interval, can be None to start from beginning
    :param end: end date of effective time interval, can be None to return all data
    :param valid_threshold: minimal ratio of series real length to entire (end-start) interval. Series dropped if
    ratio is less than threshold
    :return: tuple(log1p(series), nans, series start, series end)
    """
    df = read_x(ori_data_path, start, end)
    starts, ends = find_start_end(df.values)
    # boolean mask for bad (too short) series
    page_mask = (ends - starts) / df.shape[1] < valid_threshold
    print("Masked %d vms from %d" % (page_mask.sum(), len(df)))
    inv_mask = ~page_mask
    df = df[inv_mask]
    return df, starts[inv_mask], ends[inv_mask]

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


def run():
    parser = argparse.ArgumentParser(description='Prepare data')
    parser.add_argument('--train_data_path', default='/nfs/project/xuyixiao/zhangchao.h5'
                        , help='Path that stores the original data')
    parser.add_argument('--features_dir', default='data/vars', help='Path to store TF features')
    parser.add_argument('--valid_threshold', default=0.04, type=float, help="Series minimal length threshold (pct of data length)")
    parser.add_argument('--add_timestamp', default=288, type=int, help="Add N timestamp in a future for prediction")
    parser.add_argument('--start', default=0, type=int, help="Effective start date. Data before the start is dropped")
    parser.add_argument('--end', default=-288, type=int, help="Effective end date. Data past the end is dropped")   
    parser.add_argument('--seasonal', default=1, type=int, help='The number of low-pass filter for seasonality')
    parser.add_argument('--corr_backoffset', default=0, type=int, help='Offset for correlation calculation')
    args = parser.parse_args()

    # Get the data
    df, starts, ends = prepare_data(args.train_data_path, args.start, args.end, args.valid_threshold)

    # Our working date range
    data_start, data_end = df.columns[0], df.columns[-1]

    # We have to project some date-dependent features (day of week, etc) to the future dates for prediction
    features_end = data_end + args.add_timestamp
    print(f"start: {data_start}, end:{data_end}, features_end:{features_end}")
    features_time = features_end - data_start

    assert df.index.is_monotonic_increasing

    # daily autocorrelation
    day_autocorr = batch_autocorr(df.values, 288, starts, ends, 1.5, args.corr_backoffset)

    # weekly autocorrelation
    week_autocorr = batch_autocorr(df.values, 288 * 7, starts, ends, 2, args.corr_backoffset)

    # Normalise all the things
    day_autocorr = normalize(np.nan_to_num(day_autocorr))
    week_autocorr = normalize(np.nan_to_num(week_autocorr))

    # Make time-dependent features
    feature_time = np.arange(data_start, features_end + 1) % 288
    time_period = 288 / (2 * np.pi)
    dow_norm = feature_time / time_period  
    dow = np.stack([np.cos(dow_norm), np.sin(dow_norm)], axis=-1)
    if args.seasonal > 1:
        for k in range(2, args.seasonal + 1):
            time_period = 288 / (2 * np.pi * k)
            dow_norm = feature_time / time_period     
            dow = np.concatenate([dow, np.cos(dow_norm).reshape(-1,1), np.sin(dow_norm).reshape(-1,1)], axis=-1)

    # Assemble indices for quarterly lagged data
    lagged_ix = np.stack(lag_indexes(data_start, features_end), axis=-1)

    # Assemble final output
    tensors = dict(
        usage=df,
        lagged_ix=lagged_ix,
        vm_ix=df.index.values,
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
    VarFeeder(args.features_dir, tensors, plain)


if __name__ == '__main__':
    run()
