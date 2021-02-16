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
import idx2numpy as i2n
import numpy as np
import sys
import pickle
import random
import keras

LEARNING_RATE_CLIENT = 0.01
BATCH_SIZE = 64
EPOCHS = 10
LOAD_MODEL = False
loss_function = 'sparse_categorical_crossentropy'
optimizer = keras.optimizers.Adadelta()
DEBUG=True

def build_model():

    return keras.models.Sequential(
        [
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(62, activation='softmax'),
        ]
    )

def load_data(x):
    """
    load_data fucntion must retrun a numpy array of samples and a numpy array of classes they belong to
    If its a multi client case it must return a dict of such pairs.
    """

    data = pickle.load(open("../data/femnist/divided/femnist_" + str(x - 1) + ".pickle", "rb"))
       
    return data

def delay_function():
    return 0.0

def load_test_data():
    """
    load_data fucntion must retrun a numpy array of samples and a numpy array of classes they belong to
    If its a multi client case it must return a dict of such pairs.
    """
    data = pickle.load(open("../data/femnist/divided/femnist_test.pickle", "rb"))
    x = []
    y = []
    for i in range(len(data)):
        if DEBUG == True:
            print(data[i][0].shape)
            print(data[i][1].shape)

        x.extend(data[i][0])
        y.extend(data[i][1])

    x = np.array(x)
    y = np.array(y)
    if DEBUG == True:
        print(x.shape)
        print(y.shape)
    return x, y

process = ProcessBuilder.build_process(delay_function, multi_client=True)

process.register_process()

iterations, clients, training_set_size = process.parse_args(sys.argv)

network_model = NetworkModel(build_model, optimizer=optimizer, loss_function=loss_function, batch_size=BATCH_SIZE, averager=Averager(Averager.AveragingType.Weighted))

process.build_network(network_model)

if LOAD_MODEL == True:
    process.load_model('../models/femnist/pretrain/model.h5')

process.distribute_weights()

if process.is_client():
    process.load_dataset(load_data, training_set_size)

if process.is_server() == True:
    test_x, test_y = load_test_data()
    process.set_test_dataset(test_x, test_y)

if LOAD_MODEL == False:
    process.pretrain(rank=1, epochs=EPOCHS, iterations=2, verbose=1)

process.evaluate(verbose=0)

best_acc = 0
for x in range(iterations):
    process.distribute_weights()
    process.train(clients_in_round=clients, epochs=EPOCHS, verbose=0, drop_rate=0.01, iteration=x, max_cap=2)
    acc, _ = process.evaluate(verbose=0)
    if acc > best_acc:
        best_acc = acc
        process.save_model('../models/femnist/train/', name="model" + str(x) + ".h5")

process.synchronize(0)