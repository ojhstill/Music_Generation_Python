import os
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, pitch, interval

# File path directory to midi training data.
FILE_PATH = 'maestro/'


def get_notes(file):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    print("Parsing %s" % file)

    # Parse all midi files within directory.
    midi = converter.parse(file)
    notes_to_parse = None

    # Transpose to the same key of C major.
    k = midi.analyze('key')
    i = interval.Interval(k.tonic, pitch.Pitch('C'))
    transposed_midi = midi.transpose(i)

    try:  # File has instrument parts.
        s2 = instrument.partitionByInstrument(transposed_midi)
        notes_to_parse = s2.parts[0].recurse()
    except Exception:  # File has notes in a flat structure.
        notes_to_parse = transposed_midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes


def get_midi_dataset():

    with open('data/notes', 'rb') as filepath:
        return pickle.load(filepath)


def create_dataset():

    notes_array = []

    # Find all midi files within file path directory.
    # for root, dirs, files in os.walk(FILE_PATH):
    #     for file in files:
    #         if file.endswith('.mid'):
    #             # Extract all midi data from each midi file.
    #             notes_array.append(get_notes(root + '/' + file))
    files = [i for i in os.listdir(FILE_PATH) if i.endswith(".mid")]
    notes_array = [get_notes(FILE_PATH + i) for i in files]

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes_array, filepath)

    return notes_array


if __name__ == '__main__':
    # create_dataset()
    print(get_midi_dataset())
