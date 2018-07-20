### Env
1. tensorflow-1.4.1
2. python-3.6 / python3.5
3. pandas
4. numba, argsparse, tqdm etc.

### Data fetch
Main data path: `'/nfs/project/xuyixiao/zhangchao.h5'` -> hd

Abnormal VM path: `'/nfs/project/lihaocheng/badcase/single_rnn_mae_beyond_1000_vm_uuids'` -> List\<String\>

### Main files
> make_features.py - builds features from source data

> input_pipe.py - TF data preprocessing pipeline (assembles features into training/evaluation tensors, performs some sampling and normalisation)

> model.py - the model

> trainer.py - trains the model(s)

> hparams.py - hyperpatameter sets.

> visualize.py - visualize autocorrelation/periodic features etc.

> predict.ipynb - generates predictions and visualize the results

> predict.py - generate predictions and csv for each vm with format 'prediction/true_value' vs. timestamps

### Execute
Create data folder for further use `mkdir data`

Run `python make_features.py data/vars --split_df=0` to extract 
> previous one timestamp's cpu usgae.

> 1-day-lag, 7-day-lag's cpu usgae.

> 1-day-lag's autocorreclation, 7-day-lag's autocorrelation.

> 1-day Fourier periodic features

It will extract data and features from the input files and put them into `data/vars` as Tensorflow checkpoint. Note if `split_df = 1 or 2` the user need to 
change the destination to `data/normal_vars` or `data/bad_vars` respectively.

Run trainer: `python trainer.py --gpu=0 --split_df=0 --hparam_set=s32 --n_models=3 --name s32 --asgd_decay=0.99 --max_steps=11500 --save_from_step=10500.` This command will simultaneously train 3 models on different seeds (on a single TF graph). 

Then the model will be stored in `data/cpt/s32`

Run `python predict.py weight_path='s32' --split_df=0` to make a prediction and save the results under `data/preds`
