import pickle

train_cache = 'C:/Users/Francesco/Desktop/python_test_environment/train_cvs.cache'
eval_cache = 'C:/Users/Francesco/Desktop/python_test_environment/eval_csv.cache'

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
