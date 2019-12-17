import getopt

class Process():
    """
    Class after which Server and Client class inherit.
    """
    def __init__(self, rank, comm):
        self._rank = rank
        self._comm = comm
   
    def buildNetwork(self, networkModel):
        self._model, self._numberOfLayers = networkModel.createModel()
        self._batch_size = networkModel.getBatchSize()

    def parseArgs(self, argv):
        iterations = None
        clients = None
        training_set_size = None
        optlist, _ = getopt.getopt(argv[1:], 'i:c:t:', ['iterations=', 'clients=', 'training_set_size='])
        for currentArgument, currentValue in optlist:
            if currentArgument in ("-i", "--iterations"):
                iterations = currentValue
            elif currentArgument in ("-c", "--clients"):
                clients = currentValue
            elif currentArgument in ("-t", "--training_set_size"):
                training_set_size = int(currentValue)/100
            
        if clients == None:
            raise Exception("Missing argument clients")
        if iterations == None:
            raise Exception("Missing argument iterations")
        if training_set_size == None:
            raise Exception("Missing argument training_set_size")

        return int(iterations), int(clients), training_set_size

    def loadDataset(self, *args):
        pass

    def evaluate(self, verbose):
        return -1

    def saveModel(self, dir, name, all):
        pass

    def loadModel(self, *args):
        pass