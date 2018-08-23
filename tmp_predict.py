import tensorflow as tf
from model import Model
from input_pipe import InputPipe
from feeder import VarFeeder
from tqdm import trange
import collections
import pandas as pd
import numpy as np
from trainer import predict
from hparams import build_hparams
import hparams

paths = [p for p in tf.train.get_checkpoint_state('data/cpt/s32').all_model_checkpoint_paths]
#tf.reset_default_graph()
#preds = predict(paths, default_hparams(), back_offset=0,
#                    n_models=3, target_model=0, seed=2, batch_size=2048, asgd=True)
t_preds = []
for tm in range(3):
    tf.reset_default_graph()
    t_preds.append(predict(paths, build_hparams(hparams.params_s32), back_offset=0, predict_window=288,
                    n_models=3, target_model=tm, seed=2, batch_size=50, asgd=True))
preds=sum(t_preds) /3

# Use zero for negative predictions
preds[preds < 0.5] = 0
# Rouns predictions to nearest int
preds = np.round(preds).astype(np.int64)
print(preds.iloc[0, :-200])