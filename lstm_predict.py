"""Prediction module for the LSTM network.

This module uses the weights defined in 'lstm_model.hdf5' to create a LSTM network. The network predicts a sequence
of notes based on the original dataset and outputs to an array. The output array is passed on to 'midi_generator.py' to
convert the array to a MIDI file.
"""

# Import libraries.
import numpy as np
from keras.models import load_model

# Import modules.
import midi_reader
import midi_generator

# Settings.
TIME_STEPS = 100
OUTPUT_LENGTH = 64


def prepare_sequences(notes, pitch_names, n_vocab):
    """Prepare the sequences used by the neural network."""

    # Create dictionary map between unique notes and integers.
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))

    # Array setup.
    network_input = []
    output = []

    # Create network shape for number of time steps.
    for i in range(0, len(notes) - TIME_STEPS, 1):
        sequence_in = notes[i:i + TIME_STEPS]
        sequence_out = notes[i + TIME_STEPS]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    # Reshape and normalise the input into a format compatible with LSTM layers.
    n_patterns = len(network_input)
    normalized_input = np.reshape(network_input, (n_patterns, TIME_STEPS, 1))
    normalized_input = normalized_input / float(n_vocab)

    return network_input, normalized_input


def generate_notes(model, network_input, pitchnames, n_vocab):
    """Predict note sequence from the neural network based on trained model."""

    print('Predicting sequence...')

    # Pick a random sequence from the input as a starting point for the prediction.
    start = np.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(OUTPUT_LENGTH):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


if __name__ == '__main__':

    # Get cashed dataset.
    notes_array = midi_reader.get_midi_dataset()

    # Convert 2D array into 1D array.
    notes = [element for note in notes_array for element in note]

    # Setup LSTM network.
    pitch_names = sorted(set(item for item in notes))
    n_vocab = len(set(notes))
    network_input, normalized_input = prepare_sequences(notes, pitch_names, n_vocab)

    # Create model from trained weights.
    model = load_model('weights/lstm_model.hdf5')

    # Predict and generate MIDI output sequence.
    prediction_output = generate_notes(model, network_input, pitch_names, n_vocab)
    midi_generator.create_midi(prediction_output, 'lstm')
