"""
Experymenting
Run it with:
 mpiexec -n NUMBER_OF_CIENTS+1 python3.6 femnist.py -i 3 -c 4 
 Params:
 -i number of iterations 
 -c number of clients participating in each iteration
Provide a pickled dictionary (created by data/femnist/split_femnist.py) to each node using distribute_femnist.py.
"""

from fll import ProcessBuilder
from fll import NetworkModel
from fll import Averager
import tensorflow as tf
import idx2numpy as i2n
import numpy as np
import sys
import pickle
import random

LEARNING_RATE_CLIENT = 0.01
BATCH_SIZE = 64
EPOCHS = 10
LOAD_MODEL = True
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adadelta()

def build_model():

    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(62, activation='softmax'),
        ]
    )

def load_data():
    """
    load_data fucntion must retrun a numpy array of samples and a numpy array of classes they belong to
    If its a multi client case it must return a dict of such pairs.
    """
    data_dict = pickle.load(open("data/femnist/femnist.pickle", "rb"))
    data = []
    for key in data_dict["user_data"]:
        data.append([data_dict["user_data"][key]['x'], data_dict["user_data"][key]['y']])
       
    return data

def load_test_data(training_set_size=1.0):
    data_dict = pickle.load(open("data/femnist/femnist.pickle", "rb"))
    test_x = []
    test_y = []
    keys = list(data_dict["user_data"].keys())
    random.shuffle(keys)
    for x in range(int(len(keys)*training_set_size)):
        test_x.extend(data_dict["user_data"][keys[x]]['x'])
        test_y.extend(data_dict["user_data"][keys[x]]['y'])

    return np.array(test_x), np.array(test_y)

def delay_function():
    return 1.0

process = ProcessBuilder.build_process(delay_function, multi_client=True)

process.register_process()

iterations, clients, training_set_size = process.parse_args(sys.argv)

network_model = NetworkModel(build_model, optimizer=optimizer, loss_function=loss_function, batch_size=BATCH_SIZE, averager=Averager(Averager.AveragingType.Weighted))

process.build_network(network_model)

if LOAD_MODEL == True:
    process.load_model('models/femnist/pretrain/model.h5')

process.distribute_weights()

if process.is_client():
    process.load_dataset(load_data, training_set_size)

if process.is_server() == True:
    test_x, test_y = load_test_data(training_set_size)
    process.set_test_dataset(test_x, test_y)

if LOAD_MODEL == False:
    process.pretrain(rank=1, epochs=EPOCHS, iterations=200, verbose=1)

process.evaluate(verbose=0)

best_acc = 0
for x in range(iterations):
    process.distribute_weights()
    process.train(clients_in_round=clients, epochs=EPOCHS, verbose=0, drop_rate=0.25, iteration=x)
    acc, _ = process.evaluate(verbose=0)
    if acc > best_acc:
        best_acc = acc
        process.save_model('models/femnist/train/', name="model" + str(x) + ".h5")