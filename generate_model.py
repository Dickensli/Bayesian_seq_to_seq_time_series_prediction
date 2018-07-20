import argparse
from hparams import build_from_set, build_hparams
from trainer import train
from make_features import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the model')
    # Prepare the data
    parser.add_argument('--train_data_path', default='/nfs/project/xuyixiao/zhangchao.h5'
                        , help='Path that stores the original data')
    parser.add_argument('--valid_threshold', default=0.04, type=float, help="Series minimal length threshold (pct of data length)")
    parser.add_argument('--start', default=0, type=int, help="Effective start date. Data before the start is dropped")
    parser.add_argument('--end', default=-288, type=int, help="Effective end date. Data past the end is dropped")   
    parser.add_argument('--seasonal', default=1, type=int, help='The number of low-pass filter for seasonality')
    parser.add_argument('--corr_backoffset', default=0, type=int, help='Offset for correlation calculation')
    
    # Train the model
    parser.add_argument('--name', default='s32', help='Model name to identify different logs/checkpoints')
    parser.add_argument('--hparam_set', default='s32', help="Hyperparameters set to use (see hparams.py for available sets)")
    parser.add_argument('--n_models', default=1, type=int, help="Jointly train n models with different seeds")
    parser.add_argument('--multi_gpu', default=False,  action='store_true', help="Use multiple GPUs for multi-model training, one GPU per model")
    parser.add_argument('--seed', default=5, type=int, help="Random seed")
    parser.add_argument('--logdir', default='data/logs', help="Directory for summary logs")
    parser.add_argument('--datadir', default='data', 
                        help="Directory to store the model/TF features/other temporary variables")
    parser.add_argument('--max_epoch', type=int, default=100, help="Max number of epochs")
    parser.add_argument('--patience', type=int, default=20, help="Early stopping: stop after N epochs without improvement. Requires do_eval=True")
    parser.add_argument('--train_sampling', type=float, default=1.0, help="Sample this percent of data for training")
    parser.add_argument('--eval_sampling', type=float, default=1.0, help="Sample this percent of data for evaluation")
    parser.add_argument('--eval_memsize', type=int, default=5, help="Approximate amount of avalable memory on GPU, used for calculation of optimal evaluation batch size")
    parser.add_argument('--gpu', default=0, type=int, help='GPU instance to use')
    parser.add_argument('--gpu_allow_growth', default=False,  action='store_true', help='Allow to gradually increase GPU memory usage instead of grabbing all available memory at start')
    parser.add_argument('--save_best_model', default=True,  action='store_true', help='Save best model during training. Requires do_eval=True')
    parser.add_argument('--no_forward_split', default=True, dest='forward_split',  action='store_false', help='Use walk-forward split for model evaluation. Requires do_eval=True')
    parser.add_argument('--no_eval', default=True, dest='do_eval', action='store_false', help="Don't evaluate model quality during training")
    parser.add_argument('--no_summaries', default=True, dest='write_summaries', action='store_false', help="Don't Write Tensorflow summaries")
    parser.add_argument('--verbose', default=False, action='store_true', help='Print additional information during graph construction')
    parser.add_argument('--asgd_decay', type=float,  help="EMA decay for averaged SGD. Not use ASGD if not set")
    parser.add_argument('--no_tqdm', default=True, dest='tqdm', action='store_false', help="Don't use tqdm for status display during training")
    parser.add_argument('--max_steps', type=int, help="Stop training after max steps")
    parser.add_argument('--save_from_step', type=int, help="Save model on each evaluation (10 evals per epoch), starting from this step")
    parser.add_argument('--predict_window', default=288, type=int, help="Number of days to predict")
    args = parser.parse_args()

    param_dict = dict(vars(args))
    run(**param_dict)
    param_dict['hparams'] = build_from_set(args.hparam_set)
    del param_dict['hparam_set']
    train(**param_dict)
