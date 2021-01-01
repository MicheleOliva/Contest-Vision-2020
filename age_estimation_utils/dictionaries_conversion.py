"""
Codice per convertire i dizionari nel formato fatto da Michele nella divisione del dataset
nel formato richiesto dai data loader.
"""

# n000003/0001_01.jpg {'ID': 'n000003', 'Gender': 'm', 'Age': 46.03625477357375, 'x_min': 80, 'y_min': 19, 'width': 250, 'height': 250}

dictionary = {} # DIZIONARIO DA CONVERTIRE

francesco_dict = {}
francesco_list = []
num_samples = 0
for path in dictionary.keys():
    identity_id = dictionary[path]['ID']
    if identity_id not in francesco_dict.keys():
        francesco_dict[identity_id] = {
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
    francesco_dict[identity_id]['metadata'].append(id_data)
    if identity_id not in francesco_list:
        francesco_list.append(identity_id)
    num_samples += 1
    
    