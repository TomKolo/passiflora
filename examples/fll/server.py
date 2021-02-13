from . import Process, DEBUG
import numpy as np
import random
import tensorflow as tf
import keras
import sys
from collections import Counter

class Server(Process):
    """
    Class inheriting from Process, consist of functions performed by server to 
    cooperate with clients. There can be only one Server and it has to have rank 0.
    """
    def __init__(self, rank, size, comm, delay, device_name, multi_client):
        self.__size = size
        self.__clients_in_round = None
        self.__multi_client = multi_client
        super().__init__(rank, comm, delay, device_name)

    def pretrain(self, rank, epochs, verbose, iterations=0):
        update = None
        update = self._comm.gather(update, root=0)
        update = [x for x in update if x is not None]
        update = self.__federated_averaging(update)
        self.__apply_update(update)

    def train(self, clients_in_round, epochs, verbose, drop_rate, iteration, max_cap=1):
        if self.__clients_in_round != clients_in_round:
            self.__clients_in_round = clients_in_round
            self.__allocate()

        selected_processes = self.__rand_clients(clients_in_round, max_cap)
        self._comm.bcast(selected_processes, root=0)

        requests = []
        for i, (key, val) in enumerate(selected_processes.items()):
            requests.append(self._comm.irecv(self.__buffers[i],source=key, tag=11))
        
        update = self.__wait_for_clients(requests, drop_rate)
        update = self.__federated_averaging(update)
        self.__apply_update(update)

    def evaluate(self, verbose):
        loss, acc = self._model.evaluate(self.__test_x, self.__test_y, verbose=verbose)
        print("Accuracy: " + str(acc) + " loss: " + str(loss))
        return acc, loss

    def load_dataset(self, load_dataset_function, train_dataset_size, batch_size=None):
        dataset_x, dataset_y = load_dataset_function()
        #TODO shuffle somehow
        #dataset_x, dataset_y = shuffle(dataset_x, dataset_y, random_state=0)
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

    def build_network(self, network_model):
        return super().build_network(network_model)

    def register_process(self):
        processes=None
        processes = self._comm.gather(processes, root=0)
        print(processes)
        self._processes = {}
        for x in processes[1:]:
            if x[1] in self._processes:
                self._processes[x[1]].append(x[0])
            else:
                self._processes[x[1]] = [x[0]]

        print(self._processes)

    def save_model(self, dir="", name="model.h5", all=False):
        try:
            if all:
                path = dir + name
                self._model.save(path)
            else:
                path = dir + name
                self._model.save_weights(path)
        except ValueError:
            print("ValueError while saving model, propably caused by too low h5py version.")


    def load_model(self, path="model.h5", all=False):
        try:
            if all:
                self._model = keras.models.load_model(path)
            else:
                self._model.load_weights(path)
        except KeyError:
            print("KeyError while loading model, propably caused by too low h5py version.")

    def parse_args(self, argv):
        iterations, clients, training_set_size = super().parse_args(argv)
        return iterations, clients, training_set_size

    def set_test_dataset(self, test_dataset_x, test_dataset_y):
        self.__test_x = test_dataset_x
        self.__test_y = test_dataset_y

    def is_server(self):
        return True

    def is_client(self):
        return False

    def __federated_averaging(self, updates):
        return self._averager.calculate_average(updates, self._model, self.__multi_client)

    def __get_weights(self):
        weights = {}
        for x in range(self._number_of_layers + 1):
            weights[x] = self._model.get_layer(index=x).get_weights()
        return weights

    def __apply_update(self, update):
        try:
            for x in range(self._number_of_layers):
                self._model.get_layer(index=x).set_weights(np.add(update[x],self._model.get_layer(index=x).get_weights()))
        except IndexError as ie:
            print("Recieved weights dimentions doesn't match model " + str(ie))

    def __rand_clients(self, clients_in_round, max_cap):
        if max_cap*(self.__size - 1) < clients_in_round:
            raise Exception("This setup can't simulate that many clients in a iteration")

        full_population = [x for x in range(1, self.__size) for _ in range(max_cap)]
        clients = random.sample(full_population, k=clients_in_round)
        clients = Counter(clients)
        clients = {key:val for key, val in clients.items() if val != 0}

        if DEBUG == True:
            print("Selected clients: ")
            print(clients)

        return clients

    def __wait_for_clients(self, requests, drop_rate):
        request_recieved = 0
        data = []
        size = len(requests)
        
        while request_recieved <=  size * (1.0 - drop_rate):
            for x in range(len(requests)):
                request = requests[x].test()
                if request[0]:
                    data.append(request[1])
                    request_recieved = request_recieved + 1
                    requests.remove(requests[x])
                    break
        
        if DEBUG == True:
            print("Size of recieved: " + str(request_recieved) + " | size of canceled " + str(len(requests)))

        for x in range(len(requests)):
            requests[x].Cancel()

        return data

    def __allocate(self):
        weights = self.__get_weights()
        space = self._averager.calculate_buffer_size(weights)

        self.__buffers = []
        for _ in range(self.__clients_in_round):
            self.__buffers.append(bytearray(space))