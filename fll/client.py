from . import Process
import numpy as np

class Client(Process):
    def __init__(self, rank, comm):
        super().__init__(rank, comm)

    def pretrain(self, rank, epochs, verbose):
        update = None
        if rank == self._rank:
            self._model.fit(x=self.__data_x, y=self.__data_y, batch_size=self._batch_size, epochs=epochs, verbose=verbose)
            update = self.__calculateUpdate()

        self._comm.gather(update, root=0)
        
    def train(self,clients_in_round, epochs, verbose):
        clientsInRound = None
        clientsInRound = self._comm.bcast(clientsInRound, root=0)
        update = None
        if self._rank in clientsInRound:
            self._model.fit(x=self.__data_x, y=self.__data_y, batch_size=self._batch_size, epochs=epochs, verbose=verbose)
            update = self.__calculateUpdate()

        self._comm.gather(update, root=0)

    def __calculateUpdate(self):
        update = {}
        for x in range(self._numberOfLayers):
            update[x] = np.subtract(self._model.get_layer(index=x).get_weights(), self.__previousWeights[x])
        return update

    def distributeDataset(self):
        data = None
        data = self._comm.scatter(data, root=0)
        self.__data_x = np.array(data[0])
        self.__data_y = np.array(data[1])
    
    def distributeWeights(self):
        data = None
        data = self._comm.bcast(data, root=0)
        self.__setWeights(data)

    def buildNetwork(self, networkModel):
        return super().buildNetwork(networkModel)

    def __setWeights(self, weights):
        self.__previousWeights = weights
        try:
            for x in range(self._numberOfLayers):
                self._model.get_layer(index=x).set_weights(weights[x])
        except IndexError as ie:
            print("Recieved weights dimentions doesn't match model " + str(ie))
