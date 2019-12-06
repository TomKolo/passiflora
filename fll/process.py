from abc import ABC

class Process(ABC):
    """
    Abstract class after which Server and Client class inherit.
    """
    def __init__(self, rank, comm):
        self._rank = rank
        self._comm = comm
   
    def buildNetwork(self, networkModel):
        self._model, self._numberOfLayers = networkModel.createModel()
        self._batch_size = networkModel.getBatchSize()

    def loadDataset(self, *args):
        pass

    def evaluate(self, verbose):
        pass

    def saveModel(self, *args):
        pass

    def loadModel(self, *args):
        pass