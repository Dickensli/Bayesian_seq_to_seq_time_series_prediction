import os.path
import shutil
import sys
import numpy as np
import tensorflow as tf
from tqdm import trange
from typing import List, Tuple
import heapq
import logging
import pandas as pd
from enum import Enum

from hparams import build_from_set, build_hparams
from feeder import VarFeeder
from input_pipe import InputPipe, ModelMode, FakeSplitter, vm_features
from model import Model


log = logging.getLogger('trainer')

class Ema:
    def __init__(self, k=0.99):
        self.k = k
        self.state = None
        self.steps = 0

    def __call__(self, *args, **kwargs):
        v = args[0]
        self.steps += 1
        if self.state is None:
            self.state = v
        else:
            eff_k = min(1 - 1 / self.steps, self.k)
            self.state = eff_k * self.state + (1 - eff_k) * v
        return self.state


class Metric:
    def __init__(self, name: str, op, smoothness: float = None):
        self.name = name
        self.op = op
        self.smoother = Ema(smoothness) if smoothness else None
        self.epoch_values = []
        self.best_value = np.Inf
        self.best_step = 0
        self.last_epoch = -1
        self.improved = False
        self._top = []

    @property
    def avg_epoch(self):
        return np.mean(self.epoch_values)

    @property
    def best_epoch(self):
        return np.min(self.epoch_values)

    @property
    def last(self):
        return self.epoch_values[-1] if self.epoch_values else np.nan

    @property
    def top(self):
        return -np.mean(self._top)


    def update(self, value, epoch, step):
        if self.smoother:
            value = self.smoother(value)
        if epoch > self.last_epoch:
            self.epoch_values = []
            self.last_epoch = epoch
        self.epoch_values.append(value)
        if value < self.best_value:
            self.best_value = value
            self.best_step = step
            self.improved = True
        else:
            self.improved = False
        if len(self._top) >= 5:
            heapq.heappushpop(self._top, -value)
        else:
            heapq.heappush(self._top, -value)


class AggMetric:
    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics

    def _mean(self, fun) -> float:
        # noinspection PyTypeChecker
        return np.mean([fun(metric) for metric in self.metrics])

    @property
    def avg_epoch(self):
        return self._mean(lambda m: m.avg_epoch)

    @property
    def best_epoch(self):
        return self._mean(lambda m: m.best_epoch)

    @property
    def last(self):
        return self._mean(lambda m: m.last)

    @property
    def top(self):
        return self._mean(lambda m: m.top)

    @property
    def improved(self):
        return np.any([metric.improved for metric in self.metrics])


class DummyMetric:
    @property
    def avg_epoch(self):
        return np.nan

    @property
    def best_epoch(self):
        return np.nan

    @property
    def last(self):
        return np.nan

    @property
    def top(self):
        return np.nan

    @property
    def improved(self):
        return False

    @property
    def metrics(self):
        return []


class Stage(Enum):
    TRAIN = 0
    EVAL_FRWD = 1
    EVAL_FRWD_EMA = 2


class ModelTrainerV2:
    def __init__(self, train_model: Model, eval: List[Tuple[Stage, Model]], model_no=0,
                 patience=None, stop_metric=None, summary_writer=None):
        self.train_model = train_model
        if eval:
            self.eval_stages, self.eval_models = zip(*eval)
        else:
            self.eval_stages, self.eval_models = [], []
        self.stopped = False
        self.model_no = model_no
        self.patience = patience
        self.best_metric = np.inf
        self.bad_epochs = 0
        self.stop_metric = stop_metric
        self.summary_writer = summary_writer

        def std_metrics(model: Model, smoothness):
            return [Metric('SMAPE', model.smape, smoothness), Metric('MAE', model.mae, smoothness)]

        self._metrics = {Stage.TRAIN: std_metrics(train_model, 0.9) + [Metric('GrNorm', train_model.glob_norm)]}
        for stage, model in eval:
            self._metrics[stage] = std_metrics(model, None)
        self.dict_metrics = {key: {metric.name: metric for metric in metrics} for key, metrics in self._metrics.items()}

    def init(self, sess):
        for model in list(self.eval_models) + [self.train_model]:
            model.inp.init_iterator(sess)

    @property
    def metrics(self):
        return self._metrics

    @property
    def train_ops(self):
        model = self.train_model
        return [model.train_op]  # , model.summaries

    def metric_ops(self, key):
        return [metric.op for metric in self._metrics[key]]

    def process_metrics(self, key, run_results, epoch, step):
        metrics = self._metrics[key]
        summaries = []
        for result, metric in zip(run_results, metrics):
            metric.update(result, epoch, step)
            summaries.append(tf.Summary.Value(tag=f"{key.name}/{metric.name}_0", simple_value=result))
        return summaries

    def end_epoch(self):
        if self.stop_metric:
            best_metric = self.stop_metric(self.dict_metrics)# self.dict_metrics[Stage.EVAL_FRWD]['SMAPE'].avg_epoch
            if self.best_metric > best_metric:
                self.best_metric = best_metric
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
                if self.bad_epochs > self.patience:
                    self.stopped = True


class MultiModelTrainer:
    def __init__(self, trainers: List[ModelTrainerV2], inc_step_op,
                 misc_global_ops=None):
        self.trainers = trainers
        self.inc_step = inc_step_op
        self.global_ops = misc_global_ops or []
        self.eval_stages = trainers[0].eval_stages

    def active(self):
        return [trainer for trainer in self.trainers if not trainer.stopped]

    def _metric_step(self, stage, initial_ops, sess: tf.Session, epoch: int, step=None, repeats=1, summary_every=1):
        ops = initial_ops
        print()
        offsets, lengths = [], []
        trainers = self.active()
        for trainer in trainers:
            offsets.append(len(ops))
            metric_ops = trainer.metric_ops(stage)
            lengths.append(len(metric_ops))
            ops.extend(metric_ops)
        if repeats > 1:
            all_results = np.stack([np.array(sess.run(ops)) for _ in range(repeats)])
            results = np.mean(all_results, axis=0)
        else:
            results = sess.run(ops)
        if step is None:
            step = results[0]

        for trainer, offset, length in zip(trainers, offsets, lengths):
            chunk = results[offset: offset + length]
            summaries = trainer.process_metrics(stage, chunk, epoch, step)
            if trainer.summary_writer and step > 200 and (step % summary_every == 0):
                summary = tf.Summary(value=summaries)
                trainer.summary_writer.add_summary(summary, global_step=step)
        return results

    def train_step(self, sess: tf.Session, epoch: int):
        ops = [self.inc_step] + self.global_ops
        for trainer in self.active():
            ops.extend(trainer.train_ops)
        results = self._metric_step(Stage.TRAIN, ops, sess, epoch, summary_every=20)
        #return results[:len(self.global_ops) + 1] # step, grad_norm
        return results[0]

    def eval_step(self, sess: tf.Session, epoch: int, step, n_batches, stages:List[Stage]=None):
        target_stages = stages if stages is not None else self.eval_stages
        for stage in target_stages:
            self._metric_step(stage, [], sess, epoch, step, repeats=n_batches)

    def metric(self, stage, name):
        return AggMetric([trainer.dict_metrics[stage][name] for trainer in self.trainers])

    def end_epoch(self):
        for trainer in self.active():
            trainer.end_epoch()

    def has_active(self):
        return len(self.active())

def train(name, hparams, multi_gpu=False, n_models=1, train_completeness_threshold=0.01,
          seed=None, logdir='data/logs', datadir='data', max_epoch=1000, patience=20, train_sampling=1.0,
          eval_sampling=1.0, eval_memsize=5, gpu=0, gpu_allow_growth=False, save_best_model=True,
          forward_split=True, write_summaries=False, verbose=False, asgd_decay=None, tqdm=True,
          max_steps=None, save_from_step=None, do_eval=True, predict_window=288, **args):

    eval_k = int(round(33407 * eval_memsize / n_models))
    eval_batch_size = int(
        eval_k / (hparams.rnn_depth * hparams.encoder_rnn_layers))  
    eval_pct = 0.1
    batch_size = hparams.batch_size
    train_window = hparams.train_window
    tf.reset_default_graph()
    if seed:
        tf.set_random_seed(seed)

    with tf.device("/cpu:0"):
        inp = VarFeeder.read_vars(os.path.join(datadir, "vars"))
        splitter = FakeSplitter(vm_features(inp), 3, seed=seed, test_sampling=eval_sampling)

    real_train_vm = splitter.splits[0].train_size
    real_eval_vm = splitter.splits[0].test_size

    items_per_eval = real_eval_vm * eval_pct
    eval_batches = int(np.ceil(items_per_eval / eval_batch_size))
    steps_per_epoch = real_train_vm // batch_size
    eval_every_step = int(round(steps_per_epoch * eval_pct))
    # eval_every_step = int(round(items_per_eval * train_sampling / batch_size))

    global_step = tf.train.get_or_create_global_step()
    inc_step = tf.assign_add(global_step, 1)


    all_models: List[ModelTrainerV2] = []

    def create_model(scope, index, prefix, seed):

        with tf.variable_scope('input') as inp_scope:
            with tf.device("/cpu:0"):
                split = splitter.splits[index]
                pipe = InputPipe(datadir, inp, features=split.train_set, n_vm=split.train_size,
                                 mode=ModelMode.TRAIN, batch_size=batch_size, n_epoch=None, verbose=verbose,
                                 train_completeness_threshold=train_completeness_threshold,
                                 predict_completeness_threshold=train_completeness_threshold, train_window=train_window,
                                 predict_window=predict_window,
                                 rand_seed=seed, train_skip_first=hparams.train_skip_first,
                                 back_offset=predict_window if forward_split else 0)
                inp_scope.reuse_variables()
                if forward_split:
                    forward_eval_pipe = InputPipe(datadir, inp, features=split.test_set, n_vm=split.test_size,
                                                  mode=ModelMode.EVAL, batch_size=eval_batch_size, n_epoch=None,
                                                  verbose=verbose, predict_window=predict_window,
                                                  train_completeness_threshold=0.01, predict_completeness_threshold=0,
                                                  train_window=train_window, rand_seed=seed, runs_in_burst=eval_batches,
                                                  back_offset=predict_window)
                else:
                    forward_eval_pipe = None
        avg_sgd = asgd_decay is not None
        #asgd_decay = 0.99 if avg_sgd else None
        train_model = Model(pipe, hparams, is_train=True, graph_prefix=prefix, asgd_decay=asgd_decay, seed=seed)
        scope.reuse_variables()

        eval_stages = []
        if forward_split:
            forward_eval_model = Model(forward_eval_pipe, hparams, is_train=False, seed=seed)
            eval_stages.append((Stage.EVAL_FRWD, forward_eval_model))
            if avg_sgd:
                eval_stages.append((Stage.EVAL_FRWD_EMA, forward_eval_model))

        if write_summaries:
            summ_path = f"{logdir}/{name}_{index}"
            if os.path.exists(summ_path):
                shutil.rmtree(summ_path)
            summ_writer = tf.summary.FileWriter(summ_path)  # , graph=tf.get_default_graph()
        else:
            summ_writer = None
        if do_eval and forward_split:
            stop_metric = lambda metrics: metrics[Stage.EVAL_FRWD]['SMAPE'].avg_epoch
        else:
            stop_metric = None
        return ModelTrainerV2(train_model, eval_stages, index, patience=patience,
                              stop_metric=stop_metric,
                              summary_writer=summ_writer)


    if n_models == 1:
        device = "/cpu:0" if gpu < 0 else f"/gpu:{gpu}"
        with tf.device(device):
            scope = tf.get_variable_scope()
            all_models = [create_model(scope, 0, None, seed=seed)]
    else:
        for i in range(0, n_models):
            device = "/cpu:0" if gpu < 0 else f"/gpu:{i}" if multi_gpu else f"/gpu:{gpu}"
            with tf.device(device):
                prefix = f"m_{i}"
                with tf.variable_scope(prefix) as scope:
                    all_models.append(create_model(scope, i, prefix=prefix, seed=seed + i))
    trainer = MultiModelTrainer(all_models, inc_step)
    if save_best_model or save_from_step:
        saver_path = os.path.join(datadir, f'cpt/{name}')
        if os.path.exists(saver_path):
            shutil.rmtree(saver_path)
        os.makedirs(saver_path)
        saver = tf.train.Saver(max_to_keep=10, name='train_saver')
    else:
        saver = None
    avg_sgd = asgd_decay is not None
    if avg_sgd:
        from itertools import chain
        def ema_vars(model):
            ema = model.train_model.ema
            return {ema.average_name(v):v for v in model.train_model.ema._averages}

        ema_names = dict(chain(*[ema_vars(model).items() for model in all_models]))
        #ema_names = all_models[0].train_model.ema.variables_to_restore()
        ema_loader = tf.train.Saver(var_list=ema_names,  max_to_keep=1, name='ema_loader')
        ema_saver = tf.train.Saver(max_to_keep=1, name='ema_saver')
    else:
        ema_loader = None

    init = tf.global_variables_initializer()

    if forward_split and do_eval:
        eval_smape = trainer.metric(Stage.EVAL_FRWD, 'SMAPE')
        eval_mae = trainer.metric(Stage.EVAL_FRWD, 'MAE')
    else:
        eval_smape = DummyMetric()
        eval_mae = DummyMetric()

    train_smape = trainer.metric(Stage.TRAIN, 'SMAPE')
    train_mae = trainer.metric(Stage.TRAIN, 'MAE')
    grad_norm = trainer.metric(Stage.TRAIN, 'GrNorm')
    eval_stages = []
    ema_eval_stages = []
    if forward_split and do_eval:
        eval_stages.append(Stage.EVAL_FRWD)
        ema_eval_stages.append(Stage.EVAL_FRWD_EMA)

    # gpu_options=tf.GPUOptions(allow_growth=False),
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=tf.GPUOptions(allow_growth=gpu_allow_growth))) as sess:
        sess.run(init)
        # pipe.load_vars(sess)
        inp.restore(sess)
        for model in all_models:
            model.init(sess)
        # if beholder:
        #    visualizer = Beholder(session=sess, logdir=summ_path)
        step = 0
        prev_top = np.inf
        best_smape = np.inf
        # Contains best value (first item) and subsequent values
        best_epoch_smape = []

        for epoch in range(max_epoch):

            # n_steps = pusher.n_vm // batch_size
            if tqdm:
                tqr = trange(steps_per_epoch, desc="%2d" % (epoch + 1), leave=False)
            else:
                tqr = range(steps_per_epoch)

            for _ in tqr:
                try:
                    step = trainer.train_step(sess, epoch)
                except tf.errors.OutOfRangeError:
                    break
                    #  visualizer.update()
                if step % eval_every_step == 0:
                    if eval_stages:
                        trainer.eval_step(sess, epoch, step, eval_batches, stages=eval_stages)

                    if save_best_model and epoch > 0 and eval_smape.last < best_smape:
                        best_smape = eval_smape.last
                        saver.save(sess, os.path.join(datadir, f'cpt/{name}/cpt'), global_step=step)
                    if save_from_step and step >= save_from_step:
                        saver.save(sess, os.path.join(datadir, f'cpt/{name}/cpt'), global_step=step)

                    if avg_sgd and ema_eval_stages:
                        ema_saver.save(sess, os.path.join(datadir, 'cpt_tmp/ema'),  write_meta_graph=False)
                        # restore ema-backed vars
                        ema_loader.restore(sess, os.path.join(datadir, 'cpt_tmp/ema'))

                        trainer.eval_step(sess, epoch, step, eval_batches, stages=ema_eval_stages)
                        # restore normal vars
                        ema_saver.restore(sess, os.path.join(datadir, 'cpt_tmp/ema'))

                MAE = "%.3f/%.3f" % (eval_mae.last, train_mae.last)
                improvement = '↑' if eval_smape.improved else ' '
                SMAPE = "%s%.3f/%.3f" % (improvement, eval_smape.last,  train_smape.last)
                if tqdm:
                    tqr.set_postfix(gr=grad_norm.last, MAE=MAE, SMAPE=SMAPE)
                if not trainer.has_active() or (max_steps and step > max_steps):
                    break

            if tqdm:
                tqr.close()
            trainer.end_epoch()
            if not best_epoch_smape or eval_smape.avg_epoch < best_epoch_smape[0]:
                best_epoch_smape = [eval_smape.avg_epoch]
            else:
                best_epoch_smape.append(eval_smape.avg_epoch)

            current_top = eval_smape.top
            if prev_top > current_top:
                prev_top = current_top
                has_best_indicator = '↑'
            else:
                has_best_indicator = ' '
            status = "%2d: Best top SMAPE=%.3f%s (%s)" % (
                epoch + 1, current_top, has_best_indicator,
                ",".join(["%.3f" % m.top for m in eval_smape.metrics]))

            if trainer.has_active():
                status += ", frwd best MAE=%.3f, SMAPE=%.3f; avg MAE=%.3f, SMAPE=%.3f, %d am" % \
                          (eval_mae.best_epoch, eval_smape.best_epoch, eval_mae.avg_epoch, eval_smape.avg_epoch,
                           trainer.has_active())
                print(status, file=sys.stderr)
            else:
                print(status, file=sys.stderr)
                
                print("Early stopping!", file=sys.stderr)
                break
            if max_steps and step > max_steps:
                print("Max steps calculated", file=sys.stderr)
                break
            sys.stderr.flush()

        return np.mean(best_epoch_smape, dtype=np.float64)


def predict(checkpoints, hparams, datadir="data", return_x=False, verbose=False, 
            predict_window=288, back_offset=0, n_models=1,
            target_model=0, asgd=False, seed=1, batch_size=1024):
    with tf.variable_scope('input') as inp_scope:
        with tf.device("/cpu:0"):
            inp = VarFeeder.read_vars(os.path.join(datadir, "vars"))
            pipe = InputPipe(datadir, inp, vm_features(inp), inp.n_vm, mode=ModelMode.PREDICT, batch_size=batch_size,
                             n_epoch=1, verbose=verbose,
                             train_completeness_threshold=0.01,
                             predict_window=predict_window,
                             predict_completeness_threshold=0.0, train_window=hparams.train_window,
                             back_offset=back_offset)
    asgd_decay = 0.99 if asgd else None
    if n_models == 1:
        model = Model(pipe, hparams, is_train=False, seed=seed, asgd_decay=asgd_decay)
    else:
        models = []
        for i in range(n_models):
            prefix = f"m_{i}"
            with tf.variable_scope(prefix) as scope:
                models.append(Model(pipe, hparams, is_train=False, seed=seed, asgd_decay=asgd_decay, graph_prefix=prefix))
        model = models[target_model]

    if asgd:
        var_list = model.ema.variables_to_restore()
        prefix = f"m_{target_model}"
        for var in list(var_list.keys()):
            if var.endswith('ExponentialMovingAverage') and not var.startswith(prefix):
                del var_list[var]
    else:
        var_list = None
    saver = tf.train.Saver(name='eval_saver', var_list=var_list)
    x_buffer = []
    predictions = None
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        pipe.load_vars(sess)
        for checkpoint in checkpoints:
            pred_buffer = []
            pipe.init_iterator(sess)
            saver.restore(sess, checkpoint)
            cnt = 0
            while True:
                try:
                    if return_x:
                        pred, x, pname = sess.run([model.predictions, model.inp.true_x, model.inp.vm_ix])
                    else:
                        pred, pname = sess.run([model.predictions, model.inp.vm_ix])
                    # utf_names = [str(name, 'utf-8') for name in pname]
                    utf_names = pname
                    pred_df = pd.DataFrame(index=utf_names, data=np.expm1(pred))
                    pred_buffer.append(pred_df)
                    if return_x:
                        # noinspection PyUnboundLocalVariable
                        x_values = pd.DataFrame(index=utf_names, data=np.round(np.expm1(x)).astype(np.int64))
                        x_buffer.append(x_values)
                    newline = cnt % 80 == 0
                    if cnt > 0:
                        print('.', end='\n' if newline else '', flush=True)
                    if newline:
                        print(cnt, end='')
                    cnt += 1
                except tf.errors.OutOfRangeError:
                    print('🎉')
                    break
            cp_predictions = pd.concat(pred_buffer)
            if predictions is None:
                predictions = cp_predictions
            else:
                predictions += cp_predictions
    predictions /= len(checkpoints)
    offset = back_offset
    start_prediction = inp.data_end + 1 - offset
    end_prediction = start_prediction + predict_window - 1
    predictions.columns = np.arange(start_prediction, end_prediction + 1)
    if return_x:
        x = pd.concat(x_buffer)
        start_data = inp.data_end - hparams.train_window - 1 - back_offset
        end_data = inp.data_end - back_offset
        x.columns = np.arange(start_data, end_data - 1)
        return predictions, x
    else:
        return predictions
