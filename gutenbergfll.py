from fll import ProcessBuilder
from fll import NetworkModel
import re
import glob
import tensorflow as tf
import numpy as np

DEBUG = True
SENTENCE_LENGTH=100
BATCH_SIZE = 64
NUMBER_OF_CHARS=127-32
EMBEDING_DIM=256
NUMBER_OF_RNN=1024
LEARNING_RATE_CLIENT = 0.01
EPOCHS = 1
ITERATIONS = 10
CLIENTS_PER_ROUND = 4
TRAIN_SET_SIZE = 0.8
optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_CLIENT)
lossFunction = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def loadData():
    int2char = [chr(x) for x in range(32, 127)]
    char2int = {u: i for i, u in enumerate(int2char)}
    DATA_PATH = "data/gutenberg/"
    files = glob.glob(DATA_PATH + "*.txt")   
    x_data = []
    y_data = []
    for file in list(files):                        
        text = re.sub("[^a-zA-Z_0-9 ]",'',open(file, encoding="latin-1").read())
        text_as_int = [char2int[x] for x in text]
        number_of_sentences = int((len(text_as_int) + 1)/SENTENCE_LENGTH)
        number_of_sentences = number_of_sentences - number_of_sentences%BATCH_SIZE

        if DEBUG:
            print("There are " + str(number_of_sentences) + " sentences!")

        x_data = x_data + [text_as_int[x*SENTENCE_LENGTH: x*SENTENCE_LENGTH + 100] for x in range(number_of_sentences)]
        y_data = y_data + [text_as_int[x*SENTENCE_LENGTH + 1: x*SENTENCE_LENGTH + 101] for x in range(number_of_sentences)]

    print("There are " + str(len(x_data) ) + " sentences in total!")
    return x_data, y_data

def buildModel():
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(NUMBER_OF_CHARS, EMBEDING_DIM, batch_input_shape=[BATCH_SIZE, None]),
        tf.keras.layers.GRU(NUMBER_OF_RNN, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(NUMBER_OF_CHARS)
    ])

process = ProcessBuilder.buildProcess()
networkModel = NetworkModel(buildModel, optimizer=optimizer, lossFunction=lossFunction, batchSize=BATCH_SIZE)

process.buildNetwork(networkModel)
process.distributeWeights()

process.loadDataset(loadData, TRAIN_SET_SIZE, BATCH_SIZE)
process.distributeDataset()

process.pretrain(rank=1, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
process.evaluate(verbose=0)

for x in range(ITERATIONS):
    process.distributeWeights()
    process.train(clients_in_round=CLIENTS_PER_ROUND, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0)
    process.evaluate(verbose=0)