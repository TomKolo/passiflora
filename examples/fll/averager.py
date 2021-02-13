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

    def calculate_average(self, updates, model, multi_client, params=None):
        if self.__type == self.AveragingType.Arithmetic:
            return self.__arithmeticAverage(updates, model, multi_client)
        elif self.__type == self.AveragingType.Weighted:
            return self.__weightedAverage(updates, model, multi_client)
        elif self.__type == self.AveragingType.Performance:
            return self.__performanceAverage(updates, params)

    def __arithmeticAverage(self, updates, model, multi_client):
        sumUpdates = []
        sumWeights = 0
        for x in range(len(updates)):
            sumWeights = sumWeights + updates[x][1]

        for x in range(len(updates)):
            for y in range(len(model.layers)):
                wighted_update = np.multiply(updates[x][0][y], 1.0/sumWeights)
                if x == 0:
                    sumUpdates.append(wighted_update)
                else:
                    sumUpdates[y] = np.add(sumUpdates[y], wighted_update)

        return sumUpdates

    def __weightedAverage(self, updates, model, multi_client):
        sumUpdates = []
        sumWeights = 0
        for x in range(len(updates)):
            sumWeights = sumWeights + updates[x][1]

        for x in range(len(updates)):
            for y in range(len(model.layers)):
                if multi_client == False:
                    weight = updates[x][1]
                else:
                    weight = 1.0

                wighted_update = np.multiply(updates[x][0][y], weight/sumWeights)
                if x == 0:
                    sumUpdates.append(wighted_update)
                else:
                    sumUpdates[y] = np.add(sumUpdates[y], wighted_update)

        return sumUpdates

    def __performanceAverage(self, updates, params):
        #TODO
        return None

    def parse_update(self, update, size):
        if self.__type == self.AveragingType.Arithmetic:
            return [update, 1]
        elif self.AveragingType == self.AveragingType.Weighted:
            return [update, size]
        else: 
            return [update, 1] # placeholder for performance average

    def sum_updates(self, sum, update):
        if self.__type == self.AveragingType.Arithmetic or self.AveragingType == self.AveragingType.Performance:
            return self.__sum_arithmetic(sum, update)
        else:
            return self.__sum_weighted(sum, update)

    def __sum_arithmetic(self, sum, update):
        if sum == None:
            sum = {}
            for x in range(len(update[0])):
                sum[x] = update[0][x]

            return [sum, 1]
        else:
            for x in range(len(update[0])):
                sum[0][x] = np.add(sum[0][x], update[0][x])
            sum[1] = sum[1] + 1

            return sum


    def __sum_weighted(self, sum, update):
        if sum == None:
            sum = {}
            for x in range(len(update[0])):
                sum[x] = np.multiply(update[0][x], update[1])

            return [sum, update[1]]
        else:
            for x in range(len(update[0])):
                sum[0][x] = np.add(sum[0][x], np.multiply(update[0][x], update[1]))
            sum[1] = sum[1] + update[1]

            return sum

    def calculate_buffer_size(self, template):
        """
        Function used to calculate how much space will wieghts update take,
        needed for asynchronous communication via MPI
        """
        template = self.parse_update(template, 0)
        sum_space = 0
        if self.__type == self.AveragingType.Arithmetic or self.AveragingType == self.AveragingType.Performance:
            for x in range(len(template[0])):
                for y in range(len(template[0][x])):
                    sum_space = sum_space + sys.getsizeof(template[0][x][y])
            sum_space = sum_space + sys.getsizeof(template[0])
            sum_space = sum_space + sys.getsizeof(template[1])
            sum_space = sum_space + sys.getsizeof(template)
        elif self.__type == self.AveragingType.Weighted:
            for x in range(len(template[0])):
                for y in range(len(template[0][x])):
                    sum_space = sum_space + sys.getsizeof(template[0][x][y])
            sum_space = sum_space + sys.getsizeof(template[0])
            sum_space = sum_space + sys.getsizeof(template[1])
            sum_space = sum_space + sys.getsizeof(template)

        return sum_space


    def __print_update(self, update):
        print("Printing update")
        str(update[0][1][0][0][0])
        return None