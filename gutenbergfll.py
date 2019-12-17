"""
Implementation of next word prediction. 
Run it with:
 mpiexec -n NUMBER_OF_CIENTS+1 python3.6 gutenbergfll.py
 Params:
 -i number of iterations 
 -c number of clients
 -t training size (between 0 and 100) percentage of dataset used as training set    
"""
from fll import ProcessBuilder
from fll import NetworkModel
import re
import glob
import tensorflow as tf
import numpy as np
import sys

DEBUG = True
SENTENCE_LENGTH=100
BATCH_SIZE = 64
NUMBER_OF_CHARS=127-32
EMBEDING_DIM=256
NUMBER_OF_RNN=1024
LEARNING_RATE_CLIENT = 0.01
EPOCHS = 1
LOAD_MODEL = False
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

iterations, clients, training_set_size = process.parseArgs(sys.argv)

networkModel = NetworkModel(buildModel, optimizer=optimizer, lossFunction=lossFunction, batchSize=BATCH_SIZE)

process.buildNetwork(networkModel)

if LOAD_MODEL == True:
     process.loadModel('./models/gutenberg/pretrain/model.h5')

process.distributeWeights()

process.loadDataset(loadData, training_set_size, BATCH_SIZE)
process.distributeDataset()

if LOAD_MODEL == False:
    process.pretrain(rank=1, epochs=EPOCHS, verbose=1)

process.evaluate(verbose=0)
process.saveModel('./models/gutenberg/train/', name="model.h5", all=False)

best_acc = 0
for x in range(iterations):
    process.distributeWeights()
    process.train(clients_in_round=clients, epochs=EPOCHS, verbose=0)
    acc = process.evaluate(verbose=0)
    if acc > best_acc:
        best_acc = acc
        process.saveModel('./models/gutenberg/train/', name="model" + str(x) + ".h5", all=False)