"""
Script to play with trained rnn network for next word prediction.
Prams:
-s starting_string - string based on which network will make its predictions
-n number_of_characters - number of characters to be predicted
-p path_to_model - path to trained model
e.g. python evaluateGutenberg.py -s startingText -p models/gutenberg/pretrain/model.h5 -n 100
"""
import sys
import getopt
import tensorflow as tf
import numpy as np

NUMBER_OF_CHARS=127-32
EMBEDING_DIM=256
NUMBER_OF_RNN=1024

optlist, args = getopt.getopt(sys.argv[1:], 's:n:p:', ['starting_string=', 'number_of_characters=', 'path_to_model='])
for currentArgument, currentValue in optlist:
    if currentArgument in ("-s", "--starting_string"):
        starting_sting = currentValue
    elif currentArgument in ("-n", "--number_of_characters"):
        number_of_characters = int(currentValue)
    elif currentArgument in ("-p", "--path_to_model"):
        path = currentValue

int2char = [chr(x) for x in range(32, 127)]
char2int = {u: i for i, u in enumerate(int2char)}
idx2char = np.array(int2char)

text_generated = []
input_eval = [char2int[s] for s in starting_sting]
input_eval = tf.expand_dims(input_eval, 0)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(NUMBER_OF_CHARS, EMBEDING_DIM, batch_input_shape=[1, None]),
    tf.keras.layers.GRU(NUMBER_OF_RNN, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(NUMBER_OF_CHARS)
])

model.load_weights(path)

for i in range(number_of_characters):
    predictions = model(input_eval)

    predictions = tf.squeeze(predictions, 0)

    predictions = predictions
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

    input_eval = tf.expand_dims([predicted_id], 0)

    text_generated.append(idx2char[predicted_id])

print(starting_sting + ''.join(text_generated))