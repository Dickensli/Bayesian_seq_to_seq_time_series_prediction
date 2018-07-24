import tensorflow as tf

from feeder import VarFeeder
from enum import Enum
from typing import List, Iterable
import numpy as np
import pandas as pd
import os

from make_features import IdMap

class ModelMode(Enum):
    TRAIN = 0
    EVAL = 1,
    PREDICT = 2

class Split:
    def __init__(self, test_set: List[tf.Tensor], train_set: List[tf.Tensor], test_size: int, train_size: int):
        self.test_set = test_set
        self.train_set = train_set
        self.test_size = test_size
        self.train_size = train_size

class FakeSplitter:
    def __init__(self, tensors: List[tf.Tensor], n_splits, seed, test_sampling=1.0):
        total_vm = tensors[0].shape[0].value
        n_vm = int(round(total_vm * test_sampling))

        def mk_name(prefix, tensor):
            return prefix + '_' + tensor.name[:-2]

        def prepare_split(i):
            idx = tf.random_shuffle(tf.range(0, n_vm, dtype=tf.int32), seed + i)
            train_tensors = [tf.gather(tensor, idx, name=mk_name('shfl', tensor)) for tensor in tensors]
            if test_sampling < 1.0:
                sampled_idx = idx[:n_vm]
                test_tensors = [tf.gather(tensor, sampled_idx, name=mk_name('shfl_test', tensor)) for tensor in tensors]
            else:
                test_tensors = train_tensors
            return Split(test_tensors, train_tensors, n_vm, total_vm)

        self.splits = [prepare_split(i) for i in range(n_splits)]


class InputPipe:
    def cut(self, usage, cpu_num, start, end):
        """
        Cuts [start:end] diapason from input data
        :param usage: usage timeseries
        :param cpu_num: number of cpu timeseries
        :param start: start index
        :param end: end index
        :return: tuple (train_hits, test_hits, dow, lagged_hits)
        """
        # Pad hits to ensure we have enough array length for prediction
        usage = tf.concat([usage, tf.fill([self.predict_window], np.NaN)], axis=0)
        cropped_usage = usage[start:end]

        #cut cpu_num
        cropped_cpu_num = cpu_num[start:end]

        # cut day of week
        cropped_dow = self.inp.dow[start:end]

        # Cut lagged usage
        # gather() accepts only int32 indexes
        # Translate lag indexes to usage values
        cropped_lags = tf.cast(self.inp.lagged_ix[start:end], tf.int32)
        cropped_lags = tf.maximum(cropped_lags, 0)
        lagged_usage = tf.gather(usage, cropped_lags)

        lag_mask = cropped_lags < 0
        lag_zeros = tf.zeros_like(lagged_usage)
        lagged_usage = tf.where(lag_mask | tf.is_nan(lagged_usage), lag_zeros, lagged_usage)

        # Split for train and test
        x_usage, y_usage = tf.split(cropped_usage, [self.train_window, self.predict_window], axis=0)

        # Convert NaN to zero in for train data
        x_usage = tf.where(tf.is_nan(x_usage), tf.zeros_like(x_usage), x_usage)

        return x_usage, y_usage, cropped_cpu_num, cropped_dow, lagged_usage

    def cut_train(self, usage, cpu_num, start, *args):
        """
        Cuts a segment of time series for training. Randomly chooses starting point.
        :param usage: usage timeseries
        :param args: pass-through data, will be appended to result
        :return: result of cut() + args
        """
        n_time = self.predict_window + self.train_window
        # How much free space we have to choose starting day
        free_space = self.inp.data_time - n_time - self.back_offset - self.start_offset
        if self.verbose:
            lower_train_start = self.inp.data_start + self.start_offset
            lower_test_end = lower_train_start + n_time
            lower_test_start = lower_test_end - self.predict_window
            upper_train_start = self.inp.data_start + free_space - 1
            upper_test_end = upper_train_start + n_time
            upper_test_start = upper_test_end - self.predict_window
            print(f"Free space for training: {free_space} days.")
            print(f" Lower train {lower_train_start}, prediction {lower_test_start}..{lower_test_end}")
            print(f" Upper train {upper_train_start}, prediction {upper_test_start}..{upper_test_end}")

        self.start_offset = tf.maximum(self.start_offset, start)
        # Random starting point
        offset = tf.random_uniform((), self.start_offset, free_space, dtype=tf.int32, seed=self.rand_seed)
        end = offset + n_time
        # Cut all the things
        return self.cut(usage, cpu_num, offset, end) + args

    def cut_eval(self, usage, cpu_num, start, *args):
        """
        Cuts segment of time series for evaluation.
        Always cuts train_window + predict_window length segment beginning at start_offset point
        :param usage: usage timeseries
        :param args: pass-through data, will be appended to result
        :return: result of cut() + args
        """
        end = self.start_offset + self.train_window + self.predict_window
        return self.cut(usage, cpu_num, self.start_offset, end) + args

    def reject_filter(self, x_usage, y_usage, *args):
        """
        Rejects timeseries having too many zero datapoints (more than self.max_train_empty)
        """
        if self.verbose:
            print("max empty %d train %d predict" % (self.max_train_empty, self.max_predict_empty))
        zeros_x = tf.reduce_sum(tf.to_int32(tf.equal(x_usage, 0.0)))
        keep = zeros_x <= self.max_train_empty
        return keep

    def make_features(self, x_usage, y_usage, cpu_num, dow, lagged_usage, vm_ix,
                      day_autocorr, week_autocorr):
        """
        Main method. Assembles input data into final tensors
        """
        # Split day of week to train and test
        x_dow, y_dow = tf.split(dow, [self.train_window, self.predict_window], axis=0)

        # Split cpu_num to train and test
        x_cpu_num, y_cpu_num = tf.split(cpu_num, [self.train_window, self.predict_window], axis=0)

        # Normalize usage
        mean = tf.reduce_mean(x_usage)
        std = tf.sqrt(tf.reduce_mean(tf.squared_difference(x_usage, mean)))
        norm_x_usage = (x_usage - mean) / std
        norm_y_usage = (y_usage - mean) / std
        norm_lagged_usage = (lagged_usage - mean) / std

        # Normalize cpu num
        cpu_num_mean = tf.reduce_mean(x_cpu_num)
        norm_x_cpu_num = x_cpu_num - cpu_num_mean 
        norm_y_cpu_num = y_cpu_num - cpu_num_mean

        # Split lagged usage to train and test
        x_lagged, y_lagged = tf.split(norm_lagged_usage, [self.train_window, self.predict_window], axis=0)

        # Combine all vm features into single tensor
        stacked_features = tf.stack([day_autocorr, week_autocorr])
        flat_vm_features = stacked_features
        vm_features = tf.expand_dims(flat_vm_features, 0)

        # Train features
        x_features = tf.concat([
            # [n_days] -> [n_days, 1]
            tf.expand_dims(norm_x_usage, -1),
            x_dow,
            tf.expand_dims(norm_x_cpu_num, -1),
            x_lagged,
            # Stretch vm_features to all training days
            # [1, features] -> [n_days, features]
            tf.tile(vm_features, [self.train_window, 1])
        ], axis=1)

        # Test features
        y_features = tf.concat([
            # [n_days] -> [n_days, 1]
            y_dow,
            tf.expand_dims(norm_y_cpu_num, -1),
            y_lagged,
            # Stretch vm_features to all testing days
            # [1, features] -> [n_days, features]
            tf.tile(vm_features, [self.predict_window, 1])
        ], axis=1)

        return x_usage, x_features, norm_x_usage, x_lagged, y_usage, y_features, norm_y_usage, mean, std, flat_vm_features, vm_ix

    def __init__(self, datadir, inp: VarFeeder, features: Iterable[tf.Tensor], n_vm: int, mode: ModelMode, n_epoch=None,
                 batch_size=500, runs_in_burst=1, verbose=True, predict_window=288, train_window=28,
                 train_completeness_threshold=1, predict_completeness_threshold=1, back_offset=0,
                 train_skip_first=0, rand_seed=None):
        """
        Create data preprocessing pipeline
        :param inp: Raw input data
        :param features: Features tensors (subset of data in inp)
        :param n_vm: Total number of vms
        :param mode: Train/Predict/Eval mode selector
        :param n_epoch: Number of epochs. Generates endless data stream if None
        :param batch_size:
        :param runs_in_burst: How many batches can be consumed at short time interval (burst). Multiplicator for prefetch()
        :param verbose: Print additional information during graph construction
        :param predict_window: Number of timestamps to predict
        :param train_window: Use train_window timestamps for traning
        :param train_completeness_threshold: Percent of zero datapoints allowed in train timeseries.
        :param predict_completeness_threshold: Percent of zero datapoints allowed in test/predict timeseries.
        :param back_offset: Don't use back_offset days at the end of timeseries
        :param train_skip_first: Don't use train_skip_first days at the beginning of timeseries
        :param rand_seed:

        """
        self.n_vm = n_vm
        self.inp = inp
        self.batch_size = batch_size
        self.rand_seed = rand_seed
        self.back_offset = back_offset
        self.toId = IdMap()
        self.toId.read_pickle(os.path.join(datadir, "toId.pkl"))
        self.vm_size = len(self.toId.name2id_dict.keys())

        if verbose:
            print("Mode:%s, data days:%d, Data start:%s, data end:%s, features end:%s " % (
            mode, inp.data_time, inp.data_start, inp.data_end, inp.features_end))

        if mode == ModelMode.TRAIN:
            # reserve predict_window at the end for validation
            assert inp.data_time - predict_window > predict_window + train_window, \
                "Predict+train window length (+predict window for validation) is larger than total number of days in dataset"
            self.start_offset = train_skip_first
        elif mode == ModelMode.EVAL or mode == ModelMode.PREDICT:
            self.start_offset = inp.data_time - train_window - back_offset
            if verbose:
                train_start = inp.data_start + self.start_offset
                eval_start = train_start + train_window
                end = eval_start + predict_window - 1
                print("Train start %s, predict start %s, end %s" % (train_start, eval_start, end))
            assert self.start_offset >= 0

        self.train_window = train_window
        self.predict_window = predict_window
        self.max_train_empty = int(round(train_window * (1 - train_completeness_threshold)))
        self.max_predict_empty = int(round(predict_window * (1 - predict_completeness_threshold)))
        self.mode = mode
        self.verbose = verbose

        # Reserve more processing threads for eval/predict because of larger batches
        num_threads = 3 if mode == ModelMode.TRAIN else 6

        # Choose right cutter function for current ModelMode
        cutter = {ModelMode.TRAIN: self.cut_train, ModelMode.EVAL: self.cut_eval, ModelMode.PREDICT: self.cut_eval}
        # Create dataset, transform features and assemble batches
        root_ds = tf.data.Dataset.from_tensor_slices(tuple(features)).repeat(n_epoch)
        batch = (root_ds
                 .map(cutter[mode])
                 .filter(self.reject_filter)
                 .map(self.make_features, num_parallel_calls=num_threads)
                 .batch(batch_size)
                 .prefetch(runs_in_burst * 2)
                 )

        self.iterator = batch.make_initializable_iterator()
        it_tensors = self.iterator.get_next()

        # Assign all tensors to class variables
        self.true_x, self.time_x, self.norm_x, self.lagged_x, self.true_y, self.time_y, self.norm_y, self.norm_mean, \
        self.norm_std, self.vm_features, self.vm_ix = it_tensors

        self.encoder_features_depth = self.time_x.shape[2].value

    def load_vars(self, session):
        self.inp.restore(session)
        
    def init_iterator(self, session):
        session.run(self.iterator.initializer)


def vm_features(inp: VarFeeder):
    return (inp.usage, inp.cpu_num, inp.starts, inp.vm_ix, inp.day_autocorr, inp.week_autocorr)
