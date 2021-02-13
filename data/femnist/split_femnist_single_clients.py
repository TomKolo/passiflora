import os
import sys
import json
import numpy as np
import pickle
import getopt
import random

"""
Script dividing raw femnist data into smaller subsets of clients datasets.
Params:
-p TRAIN_SET - what percentage of original dataset is to be used as training set
-t TEST_SET - what percentage of original dataset is to be used as test set
-s SEED - seed for shuffling dataset
"""

def load_data(num, part, test_placeholer, test_part):
    with open('raw/all_data_' + str(num) + '.json') as json_file:
        part_data = json.load(json_file)
        keys = list(part_data['user_data'].keys())
        random.shuffle(keys)
        print(str(keys[4]))
        data = []
        for z in range(0, int(test_part*len(part_data['users']))):
            x = np.array(part_data['user_data'][keys[z]]['x']).reshape(-1,28,28,1)
            y = np.array(part_data['user_data'][keys[z]]['y'])
            test_placeholer.append([x, y])

        for z in range(int((1.0 - part)*len(part_data['users'])), len(part_data['users'])):
            x = np.array(part_data['user_data'][keys[z]]['x']).reshape(-1,28,28,1)
            y = np.array(part_data['user_data'][keys[z]]['y'])
            data.append([x, y])

        print("Len of test dataset: " + str(len(test_placeholer)))
    return data

def parse_params(argv):
    optlist, _ = getopt.getopt(argv[1:], 'p:t:s:', ['train_set=', 'test_set=', 'seed='],)
    train_part, test_part, random_seed = None, None, None
    for currentArgument, currentValue in optlist:
        if currentArgument in ("-p", "--train_set"):
            train_part = float(currentValue)/100
        elif currentArgument in ("-t", "--test_set"):
            test_part = float(currentValue)/100
        elif currentArgument in ("-s", "--seed"):
            random_seed = currentValue

    if train_part == None:
        raise Exception("Missing argument train_set")
    if test_part == None:
        raise Exception("Missing argument test_set")
    if random_seed != None:
        random.seed(random_seed)

    if train_part + test_part > 1:
        raise Exception("Testing dataset will have a shared samples with train dataset")

    if train_part < 0.01 or test_part < 0.01:
        raise Exception("Too small dataset sizes")

    return train_part, test_part

num_of_files = 35
train_part, test_part = parse_params(sys.argv)
last_node = 0
last_file = 0
test_placeholer = []
data = []
while last_file < num_of_files or len(data) != 0:
    if len(data) == 0:
        data.extend(load_data(last_file, train_part, test_placeholer, test_part))
        last_file = last_file + 1
    print("Saving file no. "  + str(last_node) + " consisting of one client dataset.")
    pickle.dump(data[0], open('divided_single_clients/client_' + str(last_node) + '.pickle', 'wb'))
    last_node = last_node + 1
    data = data[1:]
    print(len(data))

print("Saving test dataset consisting of " + str(len(test_placeholer)) + " client datasets.")
pickle.dump(test_placeholer, open('divided_single_clients/femnist_test.pickle', 'wb'))