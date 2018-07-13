import tensorflow as tf
from model import Model
from input_pipe import InputPipe
from feeder import VarFeeder
from make_features import read_all, read_pickle, find_start_end
from tqdm import trange
import collections
import pandas as pd
import numpy as np
from trainer import predict
from hparams import build_hparams
import hparams

def smape(true, pred):
    summ = np.abs(true) + np.abs(pred)
    smape = np.where(summ == 0, 0, np.abs(true - pred) / summ)
    return smape

def mae(true, pred):
    return np.abs(np.abs(true) - np.abs(pred))

def mean_smape(true, pred):
    raw_smape = smape(true, pred)
    masked_smape = np.ma.array(raw_smape, mask=np.isnan(raw_smape))
    return masked_smape.mean()

def mean_mae(true, pred):
    raw_mae = mae(true, pred)
    masked_mae = np.ma.array(raw_mae, mask=np.isnan(raw_mae))
    return masked_mae.mean()

def show_single(vm, scope=288, bad_case=True):
    name = preds.index[vm]
    if bad_case:
        bad_path = os.path.join('data/badcase', 'single_rnn_mae_beyond_1000_vm_uuids')
        bad_list = []
        with open(bad_path, 'r') as f:
            line = f.readline()
            while(line):
                line = line[:-1] + ".hdf5"
                if line in preds.index:
                    bad_list.append(line)
                line = f.readline()
        name = bad_list[vm]
    
    # mean mae for each row
    print(f'vm name: {name}')
    prev.loc[name, ends[vm] - scope : ends[vm]].plot(logy=True)
    (preds.loc[name, :]).plot(logy=True)
    # mean loss
    print(mean_mae(prev.loc[name, ends[vm] - 288 : ends[vm]], preds.loc[name, :]))
    
df_all = read_all()
starts, ends = find_start_end(df_all.values)
prev = df_all.apply(lambda x : np.exp(x) - 1)
paths = [p for p in tf.train.get_checkpoint_state('data/cpt/s32').all_model_checkpoint_paths]

t_preds = []
for tm in range(3):
    tf.reset_default_graph()
    t_preds.append(predict(paths, build_hparams(hparams.params_s32), back_offset=0, predict_window=288,
                    n_models=3, target_model=tm, seed=2, batch_size=50, asgd=True))
preds=sum(t_preds) /3
# mean mae
preds.index = [idx.decode('ascii') for idx in preds.index]
res = 0
for idx in preds.index:
    res += np.abs(preds.loc[idx, :] - prev.loc[idx, -288:]).sum()
print(res / (len(preds.index) * 288))