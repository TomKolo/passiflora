"""
Implementation of CNN clasifing MNIST dataset. Trainging is done by fll library using federated learning.
Run it with:
 mpiexec -n NUMBER_OF_CIENTS+1 python3.6 mnistfll.py
"""
from fll import ProcessBuilder
from fll import NetworkModel
import tensorflow as tf
import idx2numpy as i2n
import numpy as np

LEARNING_RATE_CLIENT = 0.01
BATCH_SIZE = 10
EPOCHS = 1
ITERATIONS = 10
CLIENTS_PER_ROUND = 4
TRAIN_SET_SIZE = 0.8
LOAD_MODEL = True
lossFunction = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_CLIENT)

def buildModel():
    initializer = initializer=tf.keras.initializers.GlorotUniform()
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), kernel_initializer=initializer),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=initializer),
        ]
    )

def loadData():
    train = i2n.convert_from_file('data/mnist/train-images-idx3-ubyte')
    train_labels = i2n.convert_from_file('data/mnist/train-labels-idx1-ubyte')
    test = i2n.convert_from_file('data/mnist/t10k-images-idx3-ubyte')
    test_labels = i2n.convert_from_file('data/mnist/t10k-labels-idx1-ubyte')
    train = train.reshape(train.shape[0], 28, 28, 1).astype('float32')
    test = test.reshape(test.shape[0], 28, 28, 1).astype('float32')
    return np.concatenate((train,test), axis = 0), np.concatenate((train_labels, test_labels), axis = 0)

process = ProcessBuilder.buildProcess()

networkModel = NetworkModel(buildModel, optimizer=optimizer, lossFunction=lossFunction, batchSize=BATCH_SIZE)

process.buildNetwork(networkModel)

if LOAD_MODEL == True:
    process.loadModel('./models/mnist/pretrain/model.h5')

process.distributeWeights()

process.loadDataset(loadData, TRAIN_SET_SIZE)
process.distributeDataset()

if LOAD_MODEL == False:
    process.pretrain(rank=1, epochs=EPOCHS, verbose=1)

process.evaluate(verbose=0)

best_acc = 0
for x in range(ITERATIONS):
    process.distributeWeights()
    process.train(clients_in_round=CLIENTS_PER_ROUND, epochs=EPOCHS, verbose=0)
    acc = process.evaluate(verbose=0)
    if acc > best_acc:
        process.saveModel('./models/mnist/train/', name="model" + str(x) + ".h5")