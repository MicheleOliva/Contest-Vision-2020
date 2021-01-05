# Contest Vision 2020: Gruppo 2

# Usage

## Training a new model

```
$ train.py [-c /path/to/config] [-d /path/to/checkpoint/directory]
```

The input ``csv`` must be in the format
```
/path/to/image, ID, Gender, Ground Truth, x_min, y_min, width, height
```
where ``Gender`` is currently not used, so it can be any value.

The script will generate some ``cache`` files in the current working directory, which you can substitute to the ``csv`` files in the configuration to make things run faster.


## Evaluating a model

```
$ eval.py [-m /path/to/model] /path/to/csv /path/to/dataset
```
If you don't specify a model path, the script will search for the most recent trained model in the current working directory. Also, you can point to a ``cache`` file instead of a ``csv`` file, the same used in the training phase.

## Predicting

```
$ predict.py /path/to/csv /path/to/dataset /path/to/model
```
The input ``csv`` file must be in the format
```
/path/to/image, x_min, x_max, width, height 
```
and it is not compatible with caches from the previous phases. This script does not parse the ``csv`` beforehand, thus it won't be as slow.

The script will generate a ``predictions.csv`` file in the current working directory, in the format
```
/path/to/image, predicted_age
```
with ``predicted_age`` rounded to the nearest integer.


# OLD PARAMETERS


## ``eval.py``

```python
eval_csv_path = '/content/drive/Shareddrives/Progettone/Age Estimation/caches/eval_csv.cache'
eval_dataset_root_path = '/content/eval'
```