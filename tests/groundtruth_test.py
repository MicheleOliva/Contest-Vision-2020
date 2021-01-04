import os
import pandas as pd

training_set_path = '' # path
mivia_csv_path = '' # path

print('Reading dataset folder...')
ids_folders = os.listdir(training_set_path)
print(f'Found {len(ids_folders)} identities on disk!')

print('Reading dataframe...')
mivia_csv_dataframe = pd.read_csv(mivia_csv_path, names=['Filename', 'Age'])
print(f'Dataframe count: {mivia_csv_dataframe.count()}')
print(f'Dataframe head\n: {mivia_csv_dataframe.head()}')

print('Creating auxiliary structure')
dataframe_ids = []
processed_samples = 0
for detected in mivia_csv_dataframe.iterrows(): 
    print(f'Processed {processed_samples} samples', end='\r')
    identity = detected[1]['Filename'].split('/')[0]
    if identity not in dataframe_ids:
        dataframe_ids.append(identity)
    processed_samples += 1
print('\n')
print(f'Found {len(dataframe_ids)} identities in the dataframe')

print('Checking for missing identities...')
dataframe_missing_ids = []
for disk_identity in ids_folders:
    if disk_identity not in dataframe_ids:
        dataframe_missing_ids.append(disk_identity)
    print(f'Found {len(dataframe_missing_ids)} missing ids', end='\r')
print('\n')
print(f'Dataframe is missing {len(dataframe_missing_ids)} identities')
print(f'Missing identities:\n {dataframe_missing_ids}')

