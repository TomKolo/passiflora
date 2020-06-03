import getopt

class Process():
    """
    Class after which Server and Client class inherit.
    """
    def __init__(self, rank, comm, delay, device_name):
        self._rank = rank
        self._comm = comm
        self._delay = delay
        self._device_name = device_name
   
    def build_network(self, network_model):
        self._model, self._number_of_layers = network_model.create_model()
        self._batch_size = network_model.get_batch_size()

    def parse_args(self, argv):
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

    def load_dataset(self, *args):
        pass

    def evaluate(self, verbose):
        return 0,0

    def save_model(self, dir, name, all):
        pass

    def load_model(self, *args):
        pass

    def set_test_dataset(self, test_dataset_x, test_dataset_y):
        pass