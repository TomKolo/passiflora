import os
import sys
import json
import numpy as np
import pickle
import getopt

"""
Script dividing raw femnist data into smaller subsets of clients datasets.
Params:
-n NUMBER_OF_NODES - number of nodes participating in training
-p PART - what percentage of original dataset is to be used
"""

def load_data(num, part):
    with open('raw/all_data_' + str(num) + '.json') as json_file:
        part_data = json.load(json_file)
        keys = list(part_data['user_data'].keys())
        keys.sort()
        data = []
        for z in range(int((1.0 - part)*len(part_data['users'])), len(part_data['users'])):
            x = np.array(part_data['user_data'][keys[z]]['x']).reshape(-1,28,28,1)
            y = np.array(part_data['user_data'][keys[z]]['y'])
            data.append([x, y])
    return data

def parse_params(argv):
    optlist, _ = getopt.getopt(argv[1:], 'n:p:', ['number_of_nodes=', 'part='])
    for currentArgument, currentValue in optlist:
        if currentArgument in ("-n", "--number_of_nodes"):
            number_of_nodes = int(currentValue)
        elif currentArgument in ("-p", "--part"):
            part = int(currentValue)/100

    if number_of_nodes == None:
        raise Exception("Missing argument number_of_nodes")
    if part == None:
        raise Exception("Missing argument part")

    return number_of_nodes, part

num_of_files = 35
num_of_nodes, part = parse_params(sys.argv)
node_size = int(3550*part/num_of_nodes)
last_node = 0
last_file = 0
data_placeholder = []
data = []
while last_file < num_of_files:
    if len(data) < node_size:
        data.extend(load_data(last_file, part))
        last_file = last_file + 1
    
    if len(data_placeholder) + len(data) < node_size:
        data_placeholder.extend(data)
        data = []
    elif len(data_placeholder) + len(data) >= node_size:
        end =  node_size - len(data_placeholder)
        data_placeholder.extend(data[0:end])
        data = data[end: len(data)]
        print("Saving file no. "  + str(last_node) + " consisting of " + str(len(data_placeholder)) + " client datasets.")
        pickle.dump(data_placeholder, open('divided/femnist_' + str(last_node) + '.pickle', 'wb'))
        last_node = last_node + 1
        data_placeholder = []

if len(data) + len(data_placeholder) != 0:    
    data_placeholder.extend(data)
    print("Saving file no. "  + str(last_node) + " consisting of " + str(len(data_placeholder)) + " client datasets.")
    pickle.dump(data_placeholder, open('divided/femnist_' + str(last_node) + '.pickle', 'wb'))
