from . import Process
import numpy as np
import time
from random import randint
DEBUG = True

class MultiClient(Process):
    """
    Class inheriting from Process representing multiple clients at once.
    Useful when number of clents is much greater than available nodes.
    """
    def __init__(self, rank, comm, delay, device_name):
        self.__request = None
        super().__init__(rank, comm, delay, device_name)

    def pretrain(self, rank, epochs, iterations, verbose):
        update = None
        if rank == self._rank:
            for _ in range(iterations):
                client = randint(0, len(self.__data) - 1)
                self._model.fit(x=self.__data[client][0], y=np.array(self.__data[client][1]), batch_size=self._batch_size, epochs=epochs, verbose=verbose)
            update = self.__calculate_update()

        self._comm.gather(update, root=0)
        
    def train(self, clients_in_round, epochs, verbose, drop_rate, iteration):
        selected_clients = self._comm.bcast(clients_in_round, root=0)

        if self.__request != None:
            self.__request.Cancel()
            self.__request = None

        client = randint(0, len(self.__data) - 1)
        if self._rank in selected_clients:
            self._model.fit(x=self.__data[client][0], y=self.__data[client][1], batch_size=self._batch_size, epochs=epochs, verbose=verbose)
            update = self.__calculate_update()
            self.__request = self._comm.isend(update, dest=0, tag=11)
    
    def load_dataset(self, load_dataset_function, train_dataset_size, batch_size=None):
        self.__data = load_dataset_function()

        if DEBUG:
            print("MultiClient of rank " + str(self._rank) + " has loaded dataset with " + str(len(self.__data)) + " clients")
    
    def distribute_weights(self):
        data = None
        data = self._comm.bcast(data, root=0)
        self.__set_weights(data)

    def build_network(self, network_model):
        return super().build_network(network_model)

    def register_process(self):
        processes = [self._rank, self._device_name]
        self._comm.gather(processes, root=0)

    def is_server(self):
        return False

    def is_client(self):
        return True

    def __set_weights(self, weights):
        self.__previous_weights = weights
        try:
            for x in range(self._number_of_layers):
                self._model.get_layer(index=x).set_weights(weights[x])
        except IndexError as ie:
            print("Recieved weights dimentions doesn't match model " + str(ie))

    def __calculate_update(self):
        update = {}
        for x in range(self._number_of_layers):
            update[x] = np.subtract(self._model.get_layer(index=x).get_weights(), self.__previous_weights[x])
        return update
