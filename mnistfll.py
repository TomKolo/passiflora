"""
Implementation of CNN clasifing MNIST dataset. Trainging is done by fll library using federated learning.
Run it with:
 mpiexec -n NUMBER_OF_CIENTS+1 python3.6 mnistfll.py
 Params:
 -i number of iterations 
 -c number of clients participating in each iteration
 -t training size (between 0 and 100) percentage of dataset used as training set    
"""
from fll import ProcessBuilder
from fll import NetworkModel
import tensorflow as tf
import idx2numpy as i2n
import numpy as np
import sys

LEARNING_RATE_CLIENT = 0.01
BATCH_SIZE = 10
EPOCHS = 1
LOAD_MODEL = True
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_CLIENT)

def build_model():
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

def load_data():
    train = i2n.convert_from_file('data/mnist/train-images-idx3-ubyte')
    train_labels = i2n.convert_from_file('data/mnist/train-labels-idx1-ubyte')
    test = i2n.convert_from_file('data/mnist/t10k-images-idx3-ubyte')
    test_labels = i2n.convert_from_file('data/mnist/t10k-labels-idx1-ubyte')
    train = train.reshape(train.shape[0], 28, 28, 1).astype('float32')
    test = test.reshape(test.shape[0], 28, 28, 1).astype('float32')
    return np.concatenate((train,test), axis = 0), np.concatenate((train_labels, test_labels), axis = 0)

def delay_function():
    return 1.0

process = ProcessBuilder.build_process(delay_function)

iterations, clients, training_set_size = process.parse_args(sys.argv)

network_model = NetworkModel(build_model, optimizer=optimizer, loss_function=loss_function, batch_size=BATCH_SIZE)

process.build_network(network_model)

if LOAD_MODEL == True:
    process.load_model('./models/mnist/pretrain/model.h5')

process.distribute_weights()

process.load_dataset(load_data, training_set_size)
process.distribute_dataset()

if LOAD_MODEL == False:
    process.pretrain(rank=1, epochs=EPOCHS, verbose=1)

process.evaluate(verbose=0)

best_acc = 0
for x in range(iterations):
    process.distribute_weights()
    process.train(clients_in_round=clients, epochs=EPOCHS, verbose=0)
    acc, _ = process.evaluate(verbose=0)
    if acc > best_acc:
        best_acc = acc
        process.save_model('./models/mnist/train/', name="model" + str(x) + ".h5")