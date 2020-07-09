import json
import numpy as np
import pickle
from PIL import Image

"""
Script used to divide raw femnist data into smaller subsets of clients, later distributed to clients.
Define number of nodes.
"""

def save_sample(array, file_name):
        array = np.array(array)
        array = array.reshape(28,28)
        array = array*255
        array = np.array(array, dtype=np.int8)
        img = Image.fromarray(array, 'L')
        img.save(file_name)

num_of_nodes = 8
clients_per_node = int(34/num_of_nodes)

for x in range(num_of_nodes):
    users = []
    num_samples = []
    user_data = {}
    for y in range(0, clients_per_node):
        with open('raw/all_data_' + str(x*clients_per_node+y) + '.json') as json_file:
            part_data = json.load(json_file)
            users.extend(part_data['users'])
            num_samples.extend(part_data['num_samples'])
            keys = list(part_data['user_data'].keys())
            for z in range(len(part_data['users'])):
                part_data['user_data'][keys[z]]['x'] = np.array(part_data['user_data'][keys[z]]['x']).reshape(part_data['num_samples'][z],28,28,1)
                part_data['user_data'][keys[z]]['y'] = np.array(part_data['user_data'][keys[z]]['y'])

            user_data.update(part_data['user_data'])
            if y == clients_per_node - 1:
                dictionary = {}
                dictionary['users'] = users
                dictionary['num_samples'] = num_samples
                dictionary['user_data'] = user_data
                pickle.dump(dictionary, open('femnist_' + str(x) + '.pickle', 'wb'))
                key = list(dictionary['user_data'].keys())[0]
                save_sample(dictionary['user_data'][key]['x'][0], "sample_" + str(x) + ".png")
                dictionary = None
