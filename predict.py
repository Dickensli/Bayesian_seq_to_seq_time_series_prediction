import os
import argparse
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

def predict_loss(prev, paths):
    # prev: true value
    # paths: paths to the model weights
    t_preds = []
    for tm in range(3):
        tf.reset_default_graph()
        t_preds.append(predict(paths[-1:], build_hparams(hparams.params_s32), featuresdir='data/vars', 
                               back_offset=0, predict_window=288,
                               n_models=3, target_model=tm, seed=2, batch_size=50, asgd=True))
    preds=sum(t_preds) /3
    preds.index = [idx.decode('ascii') for idx in preds.index]
    # mean mae
    res = 0
    for idx in preds.index:
        res += np.abs(preds.loc[idx, :] - prev.loc[idx, -288:]).sum()
    res /= len(preds.index) * 288
    return preds, res

def generate_result(preds, prev, save_path='data/preds'):
    for idx in preds.index:
        df = preds.T.join(prev.T, lsuffix='_preds', rsuffix='_prev').loc[:, [idx + '_preds', idx + '_prev']]
        df.columns = ['Prediction', 'Truevalue']
        df.to_csv(os.path.join(save_path, idx[:-5]), header=False, index=False)

def main():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--weight_path', default='data/cpt/s32', help='Model name to identify different logs/checkpoints')
    parser.add_argument('--featuresdir', default='data/vars', help='Path to store TF features')
    parser.add_argument('--result_path', default='data/preds', help="Path to save predictions")
    parser.add_argument('--split_df', default=0, type=int, help="Whether to split vms w.r.t. abnormal behaviour")
    args = parser.parse_args()

    df_all = read_all()
    starts, ends = find_start_end(df_all.values)
    prev = df_all.apply(lambda x : np.exp(x) - 1)
    paths = [p for p in tf.train.get_checkpoint_state(os.path.realpath(args.weight_path)).all_model_checkpoint_paths]
    preds, loss = predict_loss(prev, paths, split_df)
    print(f'Mean MAE = {loss}\n........Generate csv for each csv..........')
    generate_result(preds, prev, save_path=args.result_path)
    print('Finished!')

if __name__ == '__main__':
    main()
