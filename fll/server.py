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
        update = self.__federatedaveraging(update, [rank], 1)
        self.__applyUpdate(update)

    def train(self, clients_in_round, epochs, verbose):
        clientsInRound = random.sample(range(1, self.__size), clients_in_round)
        self._comm.bcast(clientsInRound, root=0)

        update=None
        update = self._comm.gather(update, root=0)

        update = self.__federatedaveraging(update, clientsInRound, clients_in_round)
        self.__applyUpdate(update)

    def evaluate(self, verbose):
        loss, acc = self._model.evaluate(self.__test_x, self.__test_y, verbose=verbose)
        print("Accuracy: " + str(acc) + " loss: " + str(loss))
        return acc

    def loadDataset(self, loadDatasetFunction, train_dataset_size, batch_size=None):
        dataset_x, dataset_y = loadDatasetFunction()
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
    
    def distributeDataset(self):
        self._comm.scatter(self.__data, root=0)
        self.__data = None

    def distributeWeights(self):
        data = self.__getWeights()
        self._comm.bcast(data, root=0)

    def __federatedaveraging(self, updates, clients, numberOfClients):
        sumUpdates = {}
        for i, x in enumerate(clients):
            for y in range(self._numberOfLayers):
                if i == 0:
                    sumUpdates[y] = updates[x][y]
                else:
                    sumUpdates[y] = np.add(sumUpdates[y], updates[x][y])

        for x in range(self._numberOfLayers):
            sumUpdates[x] = np.multiply(sumUpdates[x],  (1/numberOfClients))

        return sumUpdates

    def buildNetwork(self, networkModel):
        return super().buildNetwork(networkModel)

    def __getWeights(self):
        weights = {}
        for x in range(self._numberOfLayers):
            weights[x] = self._model.get_layer(index=x).get_weights()

        return weights

    def __applyUpdate(self, update):
        try:
            for x in range(self._numberOfLayers):
                self._model.get_layer(index=x).set_weights(np.add(update[x],self._model.get_layer(index=x).get_weights()))
        except IndexError as ie:
            print("Recieved weights dimentions doesn't match model " + str(ie))

    def saveModel(self, dir="", name="model.h5", all=False):
        if all:
            path = dir + name
            self._model.save(path)
        else:
            path = dir + name
            self._model.save_weights(path)

    def loadModel(self, path="model.h5", all=False):
        if all:
            self._model = tf.keras.models.load_model(path)
        else:
            self._model.load_weights(path)

    def parseArgs(self, argv):
        iterations, clients, training_set_size = super().parseArgs(argv)
        if self.__size < clients :
            raise Exception("Number of clients is smaller than number of clients participation in each iteration")
        return iterations, clients, training_set_size