
class NetworkModel:
    def __init__(self, buildModelFunction=None, optimizer=None, lossFunction=None, batchSize=None):
        self.__buildModelFunction = buildModelFunction
        self.__optimizer = optimizer
        self.__loss = lossFunction
        self.__batch_size = batchSize

    def setBuildModelFunction(self, buildModelFunction):
        self.__buildModelFunction = buildModelFunction

    def setOptimizer(self, optimizer):
        self.__optimizer = optimizer

    def getOptimizer(self):
        return self.__optimizer

    def setLossFunction(self, lossFunction):
        self.__loss = lossFunction

    def getLossFunction(self):
        return self.__loss
    
    def setBatchSize(self, batchSize):
        self.__batch_size = batchSize

    def getBatchSize(self):
        return self.__batch_size

    def createModel(self):
        model = self.__buildModelFunction()
        model.compile(loss=self.__loss, optimizer = self.__optimizer, metrics=['accuracy'])
        return model, len(model.layers)