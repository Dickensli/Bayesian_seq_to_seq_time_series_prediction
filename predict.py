import os
import argparse
import tensorflow as tf
from model import Model
from input_pipe import InputPipe
from feeder import VarFeeder
from make_features import read_all, find_start_end
from tqdm import trange
import collections
import pandas as pd
import numpy as np
from trainer import predict
from hparams import build_hparams
from make_features import IdMap
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

def gen_predict(paths, seed=5, n_models=3, predict_window=288):
    # prev: true value
    # paths: paths to the model weights
    t_preds = []
    for tm in range(3):
        tf.reset_default_graph()
        t_preds.append(predict(paths[-1:], build_hparams(hparams.params_s32), back_offset=0, predict_window=predict_window,
                        n_models=n_models, target_model=tm, seed=seed, batch_size=50, asgd=True))
    preds=sum(t_preds) /3
    return preds

def gen_loss(preds, prev):
    # mean mae
    res = 0
    for idx in range(len(preds.index)):
        res += np.abs(preds.iloc[idx, :] - prev.iloc[idx, -288:]).sum()
    res /= len(preds.index) * 288
    return res

def generate_result(preds, prev, save_path='data/preds'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for idx in preds.index:
        df = preds.T.join(prev.T, lsuffix='_preds',rsuffix='_prev').loc[:, [idx + '_preds', idx + '_prev']]
        df.columns = ['Prediction', 'Truevalue']
        df.to_csv(os.path.join(save_path, idx[:-5] + '.csv'), header=False, index=False)

def main():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--weight_path', default='data/cpt/s32', help='Model name to identify different logs/checkpoints')
    parser.add_argument('--datadir', default='data', 
                        help="Directory to store the model/TF features/other temporary variables")
    parser.add_argument('--result_path', default='data/preds', help="Path to save predictions")
    parser.add_argument('--predict_window', default=288, type=int, help="Number of timestamps to predict")
    parser.add_argument('--n_models', default=3, type=int, help="Jointly train n models with different seeds")
    parser.add_argument('--seed', default=5, type=int, help="random seed for model prediction")
    args = parser.parse_args()
    
    dfs = read_all('/nfs/isolation_project/intern/project/lihaocheng/vm/')
    df = dfs[0].iloc[:, :-288]
    starts, ends = find_start_end(df.values)
    page_mask = (ends - starts) / df.shape[1] < 0.04
    print("Masked %d vms from %d" % (page_mask.sum(), len(df)))
    inv_mask = ~page_mask
    df = dfs[0][inv_mask]
    prev = df.apply(lambda x : np.exp(x) - 1).sort_index()
   
    paths = [p for p in tf.train.get_checkpoint_state(os.path.realpath(args.weight_path)).all_model_checkpoint_paths]
    preds = gen_predict(paths, seed=args.seed, n_models=args.n_models, predict_window=args.predict_window)
    preds.index = prev.index
    loss = gen_loss(preds, prev)
    print(f'Mean MAE = {loss}\n........Generate csv for each csv..........')
    generate_result(preds, prev, save_path=args.result_path)
    print('Finished!')

if __name__ == '__main__':
    main()
