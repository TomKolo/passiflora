"""
Implementation of next word prediction. 
Run it with:
 mpiexec -n NUMBER_OF_CIENTS+1 python3.6 gutenbergfll.py
 Params:
 -i number of iterations 
 -c number of clients participating in each iteration
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
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def load_data():
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

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(NUMBER_OF_CHARS, EMBEDING_DIM, batch_input_shape=[BATCH_SIZE, None]),
        tf.keras.layers.GRU(NUMBER_OF_RNN, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(NUMBER_OF_CHARS)
    ])

process = ProcessBuilder.build_process()

iterations, clients, training_set_size = process.parse_args(sys.argv)

network_model = NetworkModel(build_model, optimizer=optimizer, loss_function=loss_function, batch_size=BATCH_SIZE)

process.build_network(network_model)

if LOAD_MODEL == True:
     process.load_model('./models/gutenberg/pretrain/model.h5')

process.distribute_weights()

process.load_dataset(load_data, training_set_size, BATCH_SIZE)
process.distribute_dataset()

if LOAD_MODEL == False:
    process.pretrain(rank=1, epochs=EPOCHS, verbose=1)

process.evaluate(verbose=0)
process.save_model('./models/gutenberg/train/', name="model.h5", all=False)

best_loss = 0
for x in range(iterations):
    process.distribute_weights()
    process.train(clients_in_round=clients, epochs=EPOCHS, verbose=0)
    _, loss = process.evaluate(verbose=0)
    if loss < best_loss:
        best_loss = loss
        process.save_model('./models/gutenberg/train/', name="model" + str(x) + ".h5", all=False)
    if DEBUG == True:
        exec(open("evaluateGutenberg.py -s When -p models/gutenberg/pretrain/model.h5 -n 100").read())
