import os
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord

# File path directory to midi training data.
FILE_PATH = 'midi_songs/'


# ''' Function to read and parse the contents of a given midi file. '''
# def read_midi(file):
#     print("Reading Music File:", file)
#
#     notes = []
#     notes_to_parse = None
#
#     # Setup midi file to parse.
#     midi = converter.parse(file)
#
#     try:
#         # Midi has instrument parts.
#         s = instrument.partitionByInstrument(midi)
#
#         # Find the piano part within the score.
#         for part in s.parts:
#             if 'Piano' in str(part):
#                 notes_to_parse = part.recurse()
#     except:
#         notes_to_parse = midi.flat.notes
#
#     # Append each element from part to an array.
#     for element in notes_to_parse:
#         if isinstance(element, note.Note):
#             # Element is an individual pitch.
#             notes.append(str(element.pitch))
#         elif isinstance(element, chord.Chord):
#             # Element is multiple pitches.
#             notes.append('.'.join(str(n) for n in element.normalOrder))
#
#     # if parts:
#     #     # File has instrument parts.
#     #     notes_to_parse = parts.parts[0].recurse()
#     # else:
#     #     # File has notes in a flat structure.
#     #     notes_to_parse = midi.flat.notes
#     #
#     # # Append each midi element to an array.
#     # for element in notes_to_parse:
#     #     if isinstance(element, note.Note):
#     #         # Element is an individual pitch.
#     #         score.append(str(element.pitch))
#     #     elif isinstance(element, chord.Chord):
#     #         # Element is multiple pitches.
#     #         score.append('.'.join(str(n) for n in element.normalOrder))
#
#     with open('data/notes', 'wb') as filepath:
#         pickle.dump(notes, filepath)
#
#     return notes

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

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def get_midi_data():
    # Find all midi files within file path directory.
    files = [i for i in os.listdir(FILE_PATH) if i.endswith('.mid')]

    # Extract all midi data from each midi file.
    notes_array = [get_notes(FILE_PATH + i) for i in files]

    return notes_array


if __name__ == "__main__":
    print(get_midi_data())
