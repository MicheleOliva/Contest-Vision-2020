import pickle

train_cache = '' # path
eval_cache = '' # path
eval_ids = '' # path
train_ids = '' # path

train_cache_dict = {}

print('reading train file')

with open(train_cache, 'rb') as train_cache_in:
    train_cache_dict = pickle.load(train_cache_in)
    print('train file read')

print('training identities: ', len(train_cache_dict['identities']))

num_samples = 0
metadata = train_cache_dict['groundtruth_metadata']
for identity in metadata.keys():
    num_samples += len(metadata[identity]['metadata'])
print('training samples: ', num_samples)

print('num samples: ', train_cache_dict['num_samples'])



eval_cache_dict = {}

print('reading eval file')

with open(eval_cache, 'rb') as eval_cache_in:
    eval_cache_dict = pickle.load(eval_cache_in)
    print('eval file read')

print('eval identities: ', len(eval_cache_dict['identities']))

num_samples = 0
metadata = eval_cache_dict['groundtruth_metadata']
for identity in metadata.keys():
    num_samples += len(metadata[identity]['metadata'])
print('training samples: ', num_samples)

print('num samples: ', eval_cache_dict['num_samples'])

print('Reading eval ids...')
eval_ids_list = []
with open(eval_ids, 'rb') as eval_ids_in:
    eval_ids_list = pickle.load(eval_ids_in)
    print('Eval ids read!')
print(f'Found {len(eval_ids_list)} identities in list')

missing_ids = 0
missing_ids_list = []
eval_gt_ids = set(eval_cache_dict['identities'])
for identity in eval_ids_list:
    if identity not in eval_gt_ids:
        missing_ids += 1
        missing_ids_list.append(identity)
print(f'Num. missing ids: {missing_ids}')
print(f'Missing ids: {missing_ids_list}')
print(f'list len: {len(missing_ids_list)}')



print('Reading train ids...')
train_ids_list = []
with open(train_ids, 'rb') as train_ids_in:
    train_ids_list = pickle.load(train_ids_in)
    print('Train ids read!')
print(f'Found {len(train_ids_list)} identities in list')

train_missing_ids = 0
train_missing_ids_list = []
train_gt_ids = set(train_cache_dict['identities'])
for identity in train_ids_list:
    if identity not in train_gt_ids:
        train_missing_ids += 1
        train_missing_ids_list.append(identity)
print(f'Num. missing ids: {train_missing_ids}')
print(f'Missing ids: {train_missing_ids_list}')

eval_ids = set(eval_cache_dict['identities'])
train_ids = set(train_cache_dict['identities'])
overlapping_ids = 0
for identity in eval_ids:
    if identity in train_ids:
        overlapping_ids += 1
print(f'Overlapping ids: {overlapping_ids}')
