"""MIDI reader module.

Reads and parses all MIDI files within the 'dataset' file path using 'music21' library representation. Data is
serialised and cached to a data file for use in model training modules.
"""

# Import libraries.
import os
import pickle
from music21 import converter, instrument, note, chord, pitch, interval

# File path directory to training dataset.
FILE_PATH = 'dataset/'


def get_notes(file):

    notes = []

    print('Parsing %s...' % file)

    # Parse all midi files within directory.
    midi = converter.parse(file)
    notes_to_parse = None

    # Transpose to the key of C major.
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
            # If element is an single notes, append pitch name.
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            # If element is a chord, append a chain of intervals from root for each note.
            notes.append('.'.join(str(n) for n in element.normalOrder))

    print('File parsed.')

    return notes


def get_midi_dataset():

    print('Loading dataset...')

    # Return cashed notes array.
    with open('data/notes', 'rb') as filepath:
        return pickle.load(filepath)


def create_dataset():

    print('Creating dataset...')

    # Find all midi files within file path directory.
    notes_array = []
    for root, dirs, files in os.walk(FILE_PATH):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                # Extract all midi data from each midi file.
                notes_array.append(get_notes(root + '/' + file))

    print('Dataset created.')

    # Cache notes array to 'notes'
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes_array, filepath)
        print('Dataset cached.')


if __name__ == '__main__':
    create_dataset()
