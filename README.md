### Env
1. tensorflow-1.4.1
2. python-3.6 / python3.5
3. pandas
4. numba, argsparse, tqdm etc.

### Data fetch
Main data path: `'/nfs/project/xuyixiao/zhangchao.h5'` -> hd

### Main files
> generate_model.py - main file to do data prepossessing and tarin the model. Dependency: make_features.py, model.py, trainer.py  
> make_features.py - builds features from source data and store in TF tensor  
> input_pipe.py - TF data preprocessing pipeline (assembles features into training/evaluation tensors, performs some sampling and normalisation)  
> model.py - the model  
> trainer.py - trains the model(s)  
> hparams.py - hyperpatameter sets.   
> predict.py - generate predictions and csv for each vm with format 'prediction/true_value' vs. timestamps  

### Execute
Choose a folder to store the weight/tensor/result information. In this demo we create `data`.

Run `python generate_model.py --gpu=-1 --name=s32 --hparam_set=s32 \  
--n_model=3 --seasonal=1 --asgd_decay=0.99 \  
--predict_window=288 --train_data_path=/nfs/project/xuyixiao/zhangchao.h5 \  
--logdir=data/logs --datadir=data --seed=5` to do:  
1. extract features:    
    > previous cpu usgae.  
    > 1-day before and 7-day before's cpu usgae.  
    > 1-day before's autocorreclation, 7-day-lag's autocorrelation.   
    > **seasonal**-rank 1-day Fourier periodic features  
    
    It will load from the original hdf5 **train_data_path** (`/nfs/project/xuyixiao/zhangchao.h5`) and extract data and features from the input files and put them into **datadir**`/vars (`data/vars`) as Tensorflow checkpoint. 

2. Train model:  
    Simultaneously train **n_model** models based on different seeds (on a single TF graph). Hyperparameters are described as **s32** from `hparam.py`.  
    Exponential moving average is adjustable with default value **asgd_decay**. Predict window is fixed to **predict_window** fot both training and prediction stages.  
    Log history will be stored in **logdir**. Multiple model generated during the training will be stored in **datadir**`/cpt/s32` (`data/cpt/s32`) and we will pick the best one for prediction.  
  
Run `python predict.py --weight_path=data/cpt/s32 --result_path=data/preds --datadir=data --n_models=3 --predict_window=288` to make a prediction and save the results under `data/preds`

1. Inference:  
    Infer the next **predict_window** cpu usage. Note **n_models** should be consistent with those used in training.  
    Use the model stored in **weight_path** (`data/cpt/s32`) and static features in **datadir**`/vars` (`data/vars`);  
    Save the prediction in **result_path** (`data/preds`)  
