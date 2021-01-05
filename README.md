# Contest Vision 2020: Gruppo 2

## Reproducing the experiment
```
$ predict.py /path/to/csv /path/to/dataset /path/to/model
```
The input ``csv`` file must be in the format
```
/path/to/image, x_min, x_max, width, height 
```

The script will generate a ``predictions.csv`` file in the current working directory, in the format
```
/path/to/image, predicted_age
```
with ``predicted_age`` rounded to the nearest integer.

An already annotated ``csv`` for predictions is avalable [HERE]()
**ADD LINK**.

## Usage

### Training a new model

```
$ train.py
```
For now, training parameters can be modified inside the script itself. Please check the commented boxes before running:

```python
#### PARAMETERS #####
some_parameter = xyz
#####################
```

The script will automatically create a new directory in the current working directory to save checkpoints in. Checkpoints will be saved on each epoch end, with a tensorboard log in the ``tensorboard`` directory.

The input ``csv`` must be in the format
```
/path/to/image, ID, Gender, Ground Truth, x_min, y_min, width, height
```
where ``Gender`` is currently not used, so it can be any value.

The script will generate some ``cache`` files in the current working directory, which you can substitute to the ``csv`` files in the configuration to make things run faster.

### Resuming training
```
$ train.py --resume
```
The script will automatically search for the most recent model, i.e. the one with the highes epoch count, and resume training. Please beware of the parameters.

### Evaluating a model

```
$ eval.py [-m /path/to/model] /path/to/csv /path/to/dataset
```

If you don't specify a model path, the script will search for the most recent trained model in the current working directory. Also, you can point to a ``cache`` file instead of a ``csv`` file, the same used in the training phase.


# OLD PARAMETERS


## ``eval.py``

```python
eval_csv_path = '/content/drive/Shareddrives/Progettone/Age Estimation/caches/eval_csv.cache'
eval_dataset_root_path = '/content/eval'
```