import os, sys
from pyspark import SparkContext
import datetime
import time
import argparse
import collections
import logging
import math
import statistics
import numpy as np
import glob
import pathlib
import re
import shutil
import h5py
import json
import subprocess

VM_PERF_FIELDS = [
    'timestamp',
    'host_id',
    'vm_id',
    'vm_blk_bps_rd_vda',
    'vm_blk_bps_wr_vda',
    'vm_blk_iops_rd_vda',
    'vm_blk_iops_wr_vda',
    'vm_blk_size_vda',
    'vm_blk_used_vda',
    'vm_cpu_num',
    'vm_cpu_usage',
    'vm_dirty_blk_size',
    'vm_dirty_mem_size',
    'vm_limit_res_flag',
    'vm_mem_size',
    'vm_mem_usage',
    'vm_migratable_flag',
    'vm_net_rx_bps',
    'vm_net_rx_bps_bw',
    'vm_net_tx_bps',
    'vm_net_tx_bps_bw',
]

DYNAMIC_FEATURE_FIELDS = [
    'vm_cpu_usage',
    'vm_mem_usage',
    'vm_blk_used_vda',
    'vm_blk_iops_rd_vda',
    'vm_blk_iops_wr_vda',
    'vm_blk_bps_rd_vda',
    'vm_blk_bps_wr_vda',
    'vm_net_rx_bps',
    'vm_net_tx_bps',
]

STATIC_FEATURE_FIELDS = [
    'vm_cpu_num',
    'vm_mem_size',
    'vm_blk_size_vda',
    'vm_net_rx_bps_bw',
    'vm_net_tx_bps_bw',
]

VmPerfRecord = collections.namedtuple('VmPerfRecord', VM_PERF_FIELDS)

def parse_token(token, name):
    if name == 'vm_id':
        return str(token)
    if name == 'host_id':
        return str(token)
    return int(token)


def parse_vm_perf_clean_data(line):
    tokens = line.rstrip('\n').split('||')
    assert len(tokens) == len(VM_PERF_FIELDS)
    tokens = [parse_token(token, name) for token, name in zip(tokens, VM_PERF_FIELDS)]
    record = VmPerfRecord._make(tokens)
    return (record, record.vm_id)

def get_vm_id_perf_clean_data(line):
    tokens = line.rstrip('\n').split('||')
    return str(tokens[2])

def extract_feature(record):
    key,dynamic_feature = extract_dynamic_data_points(record)
    static_feature = extract_static_data_points(record)
    dynamic_feature.extend(static_feature)
    return key,dynamic_feature

def extract_dynamic_data_points(record):
    timestamp_bucket = math.floor(record.timestamp / 1000 / 300) * 1000 * 300
    data_point = [getattr(record, name) for name in DYNAMIC_FEATURE_FIELDS]
    key = (record.vm_id, timestamp_bucket)
    return key, data_point

def extract_static_data_points(record):
    data_point = [getattr(record, name) for name in STATIC_FEATURE_FIELDS]
    return  data_point

# def generate_dynamic_feature_dict(key, dynamic_data_points):
#
#     def get_aggregate_values(data):
#         data = sorted(data)
#         max_val = data[-1]
#         min_val = data[0]
#         avg_val = np.mean(data)
#         second_max_val = data[-2] if len(data) >= 2 else 0
#         second_min_val = data[1] if len(data) >= 2 else 0
#         median_val = np.median(data)
#         return (max_val, min_val, avg_val, second_max_val, second_min_val, median_val)
#
#     dynamic_features = []
#     for field_idx in range(len(DYNAMIC_FEATURE_FIELDS)):
#         data = [data_point[field_idx] for data_point in dynamic_data_points]
#         aggregate_values = get_aggregate_values(data)
#         dynamic_features.extend(aggregate_values)
#     return key, np.array(dynamic_features)

def generate_feature_dict(key, dynamic_data_points):

    def get_aggregate_values(data):
        data = sorted(data)
        max_val = data[-1]
        min_val = data[0]
        avg_val = np.mean(data)
        second_max_val = data[-2] if len(data) >= 2 else 0
        second_min_val = data[1] if len(data) >= 2 else 0
        median_val = np.median(data)
        #return (max_val, min_val, avg_val, second_max_val, second_min_val, median_val)
        return [max_val]

    feature = []
    static_feature = dynamic_data_points[0][-len(STATIC_FEATURE_FIELDS):]
    dynamic_features = []
    for field_idx in range(len(DYNAMIC_FEATURE_FIELDS)):
        data = [data_point[field_idx] for data_point in dynamic_data_points]
        aggregate_values = get_aggregate_values(data)
        dynamic_features.extend(aggregate_values)

    feature.extend(dynamic_features)
    feature.extend(static_feature)
    return key, np.array(feature)

def dict_dynamic_and_static(timestamp_data_tuple_list):
    timestamp_dynamic_dict = []
    timestamp_static_dict = []
    for timestamp,data in timestamp_data_tuple_list:
        timestamp_dynamic_dict.append((timestamp,data[:-len(STATIC_FEATURE_FIELDS)]))
        timestamp_static_dict.append((timestamp, data[-len(STATIC_FEATURE_FIELDS):]))
    return dict(timestamp_dynamic_dict),dict(timestamp_static_dict)

def merge_trunks(vm_id, timestamp_data_tuple_list, begin_date, end_date, vm_user_broadcast):

    # get the timestamp list
    def generate_timestamp_bucket_list(begin_date, end_date):
        begin_time = datetime.datetime.combine(begin_date, datetime.time())
        begin_time_stamp = int(time.mktime(begin_time.timetuple()))*1000
        end_time = datetime.datetime.combine(end_date, datetime.time())
        end_time_stamp = int(time.mktime(end_time.timetuple()))*1000
        timestamp_bucket_list = list(range(begin_time_stamp, end_time_stamp, 300 * 1000))
        return timestamp_bucket_list

    timestamp_data_dict,timestamp_static_dict  = dict_dynamic_and_static(timestamp_data_tuple_list)
    timestamp_bucket_list = generate_timestamp_bucket_list(begin_date, end_date)
    user_id = vm_user_broadcast.value.get(vm_id)
    if not user_id:
        user_id = '0'*32
    MISSING_VALUE = np.zeros(len(DYNAMIC_FEATURE_FIELDS) * 6)
    MISSING_STATIC_VALUE = np.zeros(len(STATIC_FEATURE_FIELDS))
    res = []
    for timestamp_bucket in timestamp_bucket_list:
        tokens = [timestamp_bucket]
        next_timestamp_bucket = timestamp_bucket + 5 * 60 * 1000
        tokens.extend(timestamp_data_dict.get(timestamp_bucket, MISSING_VALUE))
        tokens.extend(timestamp_data_dict.get(next_timestamp_bucket - 86400 * 1000, MISSING_VALUE))
        tokens.extend(timestamp_data_dict.get(next_timestamp_bucket - 7 * 86400 * 1000, MISSING_VALUE))
        tokens.extend(sum(timestamp_data_dict.get(
            next_timestamp_bucket - days * 86400 * 1000,
            MISSING_VALUE) for days in range(1, 7 + 1)) / 7)
        tokens.extend(sum(timestamp_data_dict.get(
            next_timestamp_bucket - days * 86400 * 1000,
            MISSING_VALUE) for days in range(1, 3 + 1)) / 3)
        tokens.extend(timestamp_static_dict.get(timestamp_bucket, MISSING_STATIC_VALUE))
        res.append(tokens)
    return vm_id, res


# def merge_dynamic_trunks(vm_id, timestamp_data_tuple_list, begin_date, end_date, vm_user_broadcast):
#
#     # get the timestamp list
#     def generate_timestamp_bucket_list(begin_date, end_date):
#         begin_time = datetime.datetime.combine(begin_date, datetime.time())
#         begin_time_stamp = int(time.mktime(begin_time.timetuple()))*1000
#         end_time = datetime.datetime.combine(end_date, datetime.time())
#         end_time_stamp = int(time.mktime(end_time.timetuple()))*1000
#         timestamp_bucket_list = list(range(begin_time_stamp, end_time_stamp, 300 * 1000))
#         return timestamp_bucket_list
#
#     timestamp_data_dict = dict(timestamp_data_tuple_list)
#     timestamp_bucket_list = generate_timestamp_bucket_list(begin_date, end_date)
#     user_id = vm_user_broadcast.value.get(vm_id)
#     if not user_id:
#         user_id = '0'*32
#     MISSING_VALUE = np.zeros(len(DYNAMIC_FEATURE_FIELDS) * 6)
#     res = []
#     for timestamp_bucket in timestamp_bucket_list:
#         tokens = [timestamp_bucket]
#         next_timestamp_bucket = timestamp_bucket + 5 * 60 * 1000
#         tokens.extend(timestamp_data_dict.get(timestamp_bucket, MISSING_VALUE))
#         tokens.extend(timestamp_data_dict.get(next_timestamp_bucket - 86400 * 1000, MISSING_VALUE))
#         tokens.extend(timestamp_data_dict.get(next_timestamp_bucket - 7 * 86400 * 1000, MISSING_VALUE))
#         tokens.extend(sum(timestamp_data_dict.get(
#             next_timestamp_bucket - days * 86400 * 1000,
#             MISSING_VALUE) for days in range(1, 7 + 1)) / 7)
#         tokens.extend(sum(timestamp_data_dict.get(
#             next_timestamp_bucket - days * 86400 * 1000,
#             MISSING_VALUE) for days in range(1, 3 + 1)) / 3)
#         res.append(tokens)
#     return vm_id, res


def get_vm_perf_clean_data_path(date):
    yyyy = '{:04d}'.format(date.year)
    MM = '{:02d}'.format(date.month)
    dd = '{:02d}'.format(date.day)
    VM_PERF_CLEAN_REMOTE_PATH = '/hdp/g_swan/prod/swan/china/chishui_history/bizlog/chishui_ios_vm_perf_clean/'
    src_path = pathlib.PurePath(VM_PERF_CLEAN_REMOTE_PATH, yyyy, MM, dd, '*/*')    
    return str(src_path)


def download_from_hdfs(num):
    for i in range(num):
        get_cmd = ['hadoop', 'fs', '-get', '/user/bigdata-ms/user/adamzhangchao/xuyixiao/final/part-'+'{:05d}'.format(i), './']
        returncode = subprocess.call(get_cmd)
        sys.stdout.write('part: {}, returncode: {}'.format(i, returncode)+'\n')


def concat_to_hdf5(vm_id, vm_val, end_date):
    end_date -= datetime.timedelta(days=1)
    h5_path = '/nfs/project/xuyixiao/chishui'
    if os.path.exists(h5_path) == False:
        os.makedirs(h5_path)

    YY = str(end_date.year); MM = str("%02d" % end_date.month); DD = str("%02d" % end_date.day)
    h5_new_path = str(pathlib.PurePath(h5_path, YY, MM, DD))
    if os.path.exists(h5_new_path) == False:
        os.makedirs(h5_new_path)

    out_filename = pathlib.Path(h5_new_path, vm_id+'.hdf5')
    data = np.array(vm_val)
    with h5py.File(str(out_filename), "w") as outfile:
            outfile['data'] = data


def save_to_hdf5(num, end_date):
    for i in range(0,num):
        file_name = 'part-'+'{:05d}'.format(i)
        hdfs_file = open(file_name, 'r')
        while 1:
            line = hdfs_file.readline()
            if not line:
                break
            vm_id, vm_val = line.split('-')
            vm_val = np.array(json.loads(vm_val))
            concat_to_hdf5(vm_id, vm_val, end_date)
        hdfs_file.close()


def load_vm_uuid_to_user_uuid_dict():
    VM_UUID_TO_USER_UUID_TABLE_PATH = pathlib.Path(
        '/home/stoicxuyixiao_i/code/spark_didiyun/vm_uuid_to_user_uuid')
    result = {}
    with open(str(VM_UUID_TO_USER_UUID_TABLE_PATH), 'r') as infile:
        for line in infile:
            vm_uuid, user_uuid = line.split('\t')
            result[vm_uuid.strip()] = user_uuid.strip()
    return result



def valid_date_type(arg_date_str):
    try:
        return datetime.datetime.strptime(arg_date_str, "%Y%m%d").date()
    except ValueError:
        msg = "Given Date ({0}) not valid!".format(arg_date_str)
        msg = msg + " Expected format, yyyyMMdd!"
        raise argparse.ArgumentTypeError(msg)


def run(begin_date, end_date):
    date_range = [begin_date]
    end_date += datetime.timedelta(days=1)
    date_range = [begin_date + datetime.timedelta(days=x) for x in range(0, (end_date - begin_date).days)]
    
    sc = SparkContext(appName='chishui_spark')
    
    print ("Load files into rdd!")
    for date in date_range:
        src_path = get_vm_perf_clean_data_path(date)
        if 'src_rdd' in locals().keys():
            src_rdd = src_rdd.union(sc.textFile(src_path))
        else:
            src_rdd = sc.textFile(src_path)
    
    src_rdd = src_rdd.repartition(1000)

    print ("Parse data to record!")
    parse_rdd = src_rdd.map(lambda x: parse_vm_perf_clean_data(x))
    
    valid_vm_rdd = sc.textFile(get_vm_perf_clean_data_path(date_range[-1]))
    valid_vm_ids = valid_vm_rdd.map(lambda x: get_vm_id_perf_clean_data(x)).distinct().collect()

    parse_rdd = parse_rdd.filter(lambda x: x[1] in valid_vm_ids)    

    VM_UUID_TO_USER_UUID_DICT = load_vm_uuid_to_user_uuid_dict()

    vm_user_broadcast = sc.broadcast(VM_UUID_TO_USER_UUID_DICT)

    require_dynamic_feature=True

    if require_dynamic_feature:
        print ("Feature extraction start")
        # key = (record.vm_id, timestamp_bucket), data_point[9 dynamic feature] one time
        rdd1 = parse_rdd.map(lambda x: extract_feature(x[0]))
        print ("Feature extraction success")
        #print (rdd1.take(2))
        # [(record.vm_id, timestamp_bucket),[num1,num2,num3,num4...]]
        rdd2 = rdd1.groupByKey().mapValues(list)
        # [key,np.array(dynamic_features)(statistics)]
        rdd3 = rdd2.map(lambda x: generate_feature_dict(x[0], x[1]))
        #[record.vm_id,(timestamp_bucket,np.array(dynamic_features)(statistics)]
        rdd4 = rdd3.map(lambda x: (x[0][0], (x[0][1], x[1])))
        print ("Feature select success")
        #print (rdd4.take(2))
        # [record.vm_id,[time1,time2,time3....]]time1=(timestamp_bucket,np.array(dynamic_features)(statistics)
        rdd5 = rdd4.groupByKey().mapValues(list)
        #
        #print rdd5.take(2)
        rdd6 = rdd5.map(lambda x: merge_trunks(x[0], x[1], begin_date, end_date, vm_user_broadcast))
        #print rdd6.take(2)
        dynamic_feature_rdd = rdd6.map(lambda x: str(x[0])+'-'+json.dumps(x[1]))      
        #print rdd6.take(2)
        print ("Save data to hdfs!")        
        dynamic_feature_rdd.repartition(10).saveAsTextFile('/user/bigdata-ms/user/adamzhangchao/xuyixiao/final')

        print ("Download data from hdfs!")
        download_from_hdfs(10)
        
        print ("Save data to hdf5")
        save_to_hdf5(10, end_date)

    sc.stop()


if __name__ == '__main__':
    print (datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    begin_date = valid_date_type(sys.argv[1]); end_date = valid_date_type(sys.argv[2])
    assert begin_date <= end_date
    run(begin_date, end_date)
    print (datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
