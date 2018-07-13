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
Run `ipython make_features.py data/vars` to extract 
> one timestamp before cpu usgae.

> 1-day lag, 7-day lag cpu usgae.

> 1-day lag autocorreclation, 7-day lag autocorrelation.

> 1-day Fourier periodic features

It will extract data and features from the input files and put them into data/vars as Tensorflow checkpoint.

Run trainer: `python trainer.py --hparam_set=s32 --n_models=3 --name s32 --asgd_decay=0.99 --max_steps=11500 --save_from_step=10500.` This command will simultaneously train 3 models on different seeds (on a single TF graph). 

Run `python predict.py weight_path='s32'` to make a prediction and save the results under `data/preds`
