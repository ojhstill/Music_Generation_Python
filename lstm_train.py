"""Training module for the LSTM network.

This module retrieves the midi file data and feeds it to the LSTM neural network for training. The final network
weightings are saved and output to 'lstm_model.hdf5' to be used for model prediction.
"""

# Import libraries.
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

# Import modules.
import midi_reader

# Settings.
TIME_STEPS = 100


def prepare_training_sequences(notes, n_vocab):
    """Prepare the sequences used by the neural network."""

    print('Preparing training sequence...')

    # Create a sorted list of all individual elements.
    pitch_names = sorted(set(item for item in notes))

    # Create dictionary map between unique notes and integers.
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))

    network_input = []
    network_output = []

    # Create network sequences for number of time steps.
    for i in range(0, len(notes) - TIME_STEPS, 1):
        sequence_in = notes[i:i + TIME_STEPS]
        sequence_out = notes[i + TIME_STEPS]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    # Reshape and normalise the input into a format compatible with LSTM layers.
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, TIME_STEPS, 1))
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    print('Training sequence prepared.')

    return network_input, network_output


def lstm(network_input, n_vocab):
    """LSTM model architecture."""

    # Create LSTM network structure.
    model = Sequential()
    model.add(LSTM(
        128,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(128, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(128))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def train(model, network_input, network_output):

    filepath = 'weights/lstm_model.hdf5'
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=50, batch_size=128, callbacks=callbacks_list)


if __name__ == '__main__':

    # Get cashed dataset.
    notes_array = midi_reader.get_midi_dataset()

    # Convert 2D array into 1D array.
    notes = [element for note in notes_array for element in note]

    # Setup LSTM network.
    n_vocab = len(set(notes))
    network_input, network_output = prepare_training_sequences(notes, n_vocab)
    model = lstm(network_input, n_vocab)

    # Train model.
    train(model, network_input, network_output)
