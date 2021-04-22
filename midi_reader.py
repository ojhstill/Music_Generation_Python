import os
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord

# File path directory to midi training data.
FILE_PATH = 'maestro/2018'


def get_notes(file):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    print("Parsing %s" % file)

    # Parse all midi files within directory.
    midi = converter.parse(file)
    notes_to_parse = None

    try:  # File has instrument parts.
        s2 = instrument.partitionByInstrument(midi)
        notes_to_parse = s2.parts[0].recurse()
    except Exception:  # File has notes in a flat structure.
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes


def get_midi_data():

    notes_array = []

    # Find all midi files within file path directory.
    for root, dirs, files in os.walk(FILE_PATH):
        for file in files:
            if file.endswith(".mid"):
                # Extract all midi data from each midi file.
                notes_array.append(get_notes(root + '/' + file))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes_array, filepath)

    return notes_array


if __name__ == "__main__":
    print(get_midi_data())
