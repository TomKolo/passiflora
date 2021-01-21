import sys
import enum
import numpy as np

class Averager():
    """
    Class used to calculate various types of update averages
    """
    class AveragingType(enum.Enum):
        Arithmetic = 1
        Weighted = 2
        Performance = 3

    def __init__(self, type=AveragingType.Arithmetic):
        self.__type = type

    def get_update(self, update, size):
        if self.__type == self.AveragingType.Arithmetic:
            return update
        elif self.__type == self.AveragingType.Weighted:
            return [update, size]
        elif self.__type == self.AveragingType.Performance:
            return update

    def calculate_average(self, updates, model, params=None):
        if self.__type == self.AveragingType.Arithmetic:
            return self.arithmeticAverage(updates, model)
        elif self.__type == self.AveragingType.Weighted:
            return self.weightedAverage(updates, model)
        elif self.__type == self.AveragingType.Performance:
            return self.performanceAverage(updates, params)

    def arithmeticAverage(self, updates, model):
        sumUpdates = []
        for x in range(len(updates)):
            for y in range(len(model.layers)):
                if x == 0:
                    sumUpdates.append(updates[x][y])
                else:
                    sumUpdates[y] = np.add(sumUpdates[y], updates[x][y])

        for x in range(len(model.layers)):
            sumUpdates[x] = np.multiply(sumUpdates[x],  (1/len(updates)))

        return sumUpdates

    def weightedAverage(self, data, model):
        sumUpdates = []
        sumWeights = 0
        for x in range(len(data)):
            sumWeights = sumWeights + data[x][1]

        for x in range(len(data)):
            for y in range(len(model.layers)):
                wighted_update = np.multiply(data[x][0][y], data[x][1]/sumWeights)
                if x == 0:
                    sumUpdates.append(wighted_update)
                else:
                    sumUpdates[y] = np.add(sumUpdates[y], wighted_update)

        return sumUpdates

    def performanceAverage(self, updates, params):
        #TODO
        return None

    def parse_update(self, update, size):
        if self.__type == self.AveragingType.Arithmetic or self.AveragingType == self.AveragingType.Performance:
            return update
        else: #self.AveragingType == self.AveragingType.Weighted:
            return [update, size]

    def calculate_buffer_size(self, template):
        """
        Function used to calculate how much space will wieghts update take,
        needed for asynchronous communication via MPI
        """
        template = self.parse_update(template, 0)
        sum_space = 0
        if self.__type == self.AveragingType.Arithmetic or self.AveragingType == self.AveragingType.Performance:
            for x in range(len(template)):
                for y in range(len(template[x])):
                    sum_space = sum_space + sys.getsizeof(template[x][y])
            sum_space = sum_space + sys.getsizeof(template)
        elif self.__type == self.AveragingType.Weighted:
            for x in range(len(template[0])):
                for y in range(len(template[0][x])):
                    sum_space = sum_space + sys.getsizeof(template[0][x][y])
            sum_space = sum_space + sys.getsizeof(template[0])
            sum_space = sum_space + sys.getsizeof(template[1])
            sum_space = sum_space + sys.getsizeof(template)

        return sum_space