"""
Codice per convertire i dizionari nel formato utilizzato nella divisione del dataset
nel formato richiesto dai data loader.
"""

# n000003/0001_01.jpg {'ID': 'n000003', 'Gender': 'm', 'Age': 46.03625477357375, 'x_min': 80, 'y_min': 19, 'width': 250, 'height': 250}

import pickle

eval_dict_path = '' # path

print('Reading eval dict dump...')
with open(eval_dict_path, 'rb') as eval_dict_in:
    eval_dict_dump = pickle.load(eval_dict_in)
    print('Eval dict dump read!')

dictionary = eval_dict_dump # DIZIONARIO DA CONVERTIRE

new_dict = {}
new_list = []
num_samples = 0
for path in dictionary.keys():
    identity_id = dictionary[path]['ID']
    if identity_id not in new_dict.keys():
        new_dict[identity_id] = {
            'index': 0,
            'metadata': []
        }
    id_data = {
        'age': dictionary[path]['Age'],
        'roi':{
            'upper_left_x': dictionary[path]['x_min'],
            'upper_left_y': dictionary[path]['y_min'],
            'width': dictionary[path]['width'],
            'height': dictionary[path]['height']
        },
        'path': path
    }
    new_dict[identity_id]['metadata'].append(id_data)
    if identity_id not in new_list:
        new_list.append(identity_id)
    num_samples += 1
    
print(f'Num samples: {num_samples}')
print(f'Num ids: {len(new_list)}')
print(f'Num dictionary keys: {len(new_dict.keys())}')