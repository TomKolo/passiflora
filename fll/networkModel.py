
class NetworkModel:
    """
    Simple class representing a prototype of neural network.
    """
    def __init__(self, __build_model_function=None, optimizer=None, loss_function=None, batch_size=None):
        self.__build_model_function = __build_model_function
        self.__optimizer = optimizer
        self.__loss = loss_function
        self.__batch_size = batch_size

    def set_build_model_function(self, __build_model_function):
        self.__build_model_function = __build_model_function

    def set_optimizer(self, optimizer):
        self.__optimizer = optimizer

    def get_optimizer(self):
        return self.__optimizer

    def set_loss_function(self, loss_function):
        self.__loss = loss_function

    def get_loss_function(self):
        return self.__loss
    
    def set_batch_size(self, batch_size):
        self.__batch_size = batch_size

    def get_batch_size(self):
        return self.__batch_size

    def create_model(self):
        model = self.__build_model_function()
        model.compile(loss=self.__loss, optimizer = self.__optimizer, metrics=['accuracy'])
        return model, len(model.layers)