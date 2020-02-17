from . import Process
import numpy as np
import random
from sklearn.utils import shuffle
import tensorflow as tf

DEBUG = True
class Server(Process):
    def __init__(self, rank, size, comm):
        self.__size = size
        super().__init__(rank, comm)

    def pretrain(self, rank, epochs, verbose):
        update = None
        update = self._comm.gather(update, root=0)
        update = self.__federated_averaging(update, [rank], 1)
        self.__apply_update(update)

    def train(self, clients_in_round, epochs, verbose):
        selected_clients = random.sample(range(1, self.__size), clients_in_round)
        self._comm.bcast(selected_clients, root=0)

        update=None
        update = self._comm.gather(update, root=0)

        update = self.__federated_averaging(update, selected_clients, clients_in_round)
        self.__apply_update(update)

    def evaluate(self, verbose):
        loss, acc = self._model.evaluate(self.__test_x, self.__test_y, verbose=verbose)
        print("Accuracy: " + str(acc) + " loss: " + str(loss))
        return acc, loss

    def load_dataset(self, load_dataset_function, train_dataset_size, batch_size=None):
        dataset_x, dataset_y = load_dataset_function()
        dataset_x, dataset_y = shuffle(dataset_x, dataset_y, random_state=0)
        dataset_size = len(dataset_x)
        self.__client_set_size = int(dataset_size*train_dataset_size/(self.__size - 1))
        self.__test_set_size = int(dataset_size - dataset_size*train_dataset_size)

        if batch_size != None:
            self.__client_set_size = self.__client_set_size  - (self.__client_set_size % batch_size)
            self.__test_set_size = self.__test_set_size - (self.__test_set_size % batch_size)
            
        train_x = [dataset_x[i] for i in range(0, int(dataset_size*train_dataset_size))]
        train_y = [dataset_y[i] for i in range(0, int(dataset_size*train_dataset_size))]
        self.__test_x = np.array([dataset_x[int(dataset_size*train_dataset_size) + i] for i in range(0, self.__test_set_size)])
        self.__test_y = np.array([dataset_y[int(dataset_size*train_dataset_size) + i] for i in range(0, self.__test_set_size)])

        train_x_divided = [[]] + [train_x[self.__client_set_size * x: self.__client_set_size * (x + 1)] for x in range(0, self.__size - 1)]
        train_y_divided = [[]] + [train_y[self.__client_set_size * x: self.__client_set_size * (x + 1)] for x in range(0, self.__size - 1)]

        if DEBUG:
            print("There are " + str(self.__size - 1) + " clients with datasets of size " + str(self.__client_set_size) + " and testing set of size " + str(len(self.__test_x)))

        self.__data = list(zip(train_x_divided, train_y_divided))
    
    def distribute_dataset(self):
        self._comm.scatter(self.__data, root=0)
        self.__data = None

    def distribute_weights(self):
        data = self.__get_weights()
        self._comm.bcast(data, root=0)

    def __federated_averaging(self, updates, clients, number_of_clients):
        sumUpdates = {}
        for i, x in enumerate(clients):
            for y in range(self._number_of_layers):
                if i == 0:
                    sumUpdates[y] = updates[x][y]
                else:
                    sumUpdates[y] = np.add(sumUpdates[y], updates[x][y])

        for x in range(self._number_of_layers):
            sumUpdates[x] = np.multiply(sumUpdates[x],  (1/number_of_clients))

        return sumUpdates

    def build_network(self, network_model):
        return super().build_network(network_model)

    def __get_weights(self):
        weights = {}
        for x in range(self._number_of_layers):
            weights[x] = self._model.get_layer(index=x).get_weights()

        return weights

    def __apply_update(self, update):
        try:
            for x in range(self._number_of_layers):
                self._model.get_layer(index=x).set_weights(np.add(update[x],self._model.get_layer(index=x).get_weights()))
        except IndexError as ie:
            print("Recieved weights dimentions doesn't match model " + str(ie))

    def save_model(self, dir="", name="model.h5", all=False):
        if all:
            path = dir + name
            self._model.save(path)
        else:
            path = dir + name
            self._model.save_weights(path)

    def load_model(self, path="model.h5", all=False):
        if all:
            self._model = tf.keras.models.load_model(path)
        else:
            self._model.load_weights(path)

    def parse_args(self, argv):
        iterations, clients, training_set_size = super().parse_args(argv)
        if self.__size < clients :
            raise Exception("Number of clients is smaller than number of clients participation in each iteration")
        return iterations, clients, training_set_size