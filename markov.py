"""Module for the Markov chain model.

This module retrieves the midi file data and constructs a weighted graph structure using unique notes and note
transitions. After construction, a random starting node is selected based on note occurrence in the score. Successive
nodes are randomly selected from the original notes neighbour array and are weighted to their transition occurrence in
the score. The output array is passed on to 'midi_generator.py' to convert the array to a MIDI file.
"""

# Import libraries.
import random
from music21 import stream, note

# Import modules.
import midi_reader
import midi_generator

# Settings.
OUTPUT_LENGTH = 64

# Array setup.
data_graph = []
markov_out = []
s = stream.Stream()

# Get parsed midi data.
notes_array = midi_reader.get_midi_dataset()

# Convert 2D array into 1D array.
notes = [element for note in notes_array for element in note]

# Create a sorted list of all individual elements.
pitch_names = sorted(set(item for item in notes))

print('Creating Markov graph...')

# Setup a node for each individual element with element count and neighbour array.
for pitch in pitch_names:
    node = {
        'pitch': pitch,
        'count': notes.count(pitch),
        'nbs': []
    }
    # Append onto data out.
    data_graph.append(node)

# Iterate through each midi element in the linear score.
for i in range(len(notes)-1):
    current_pitch = notes[i]
    next_pitch = notes[i+1]

    # Find the index within node array, else '-1'.
    pitch_index = next((i for i, item in enumerate(data_graph) if item['pitch'] == current_pitch), -1)
    # Get the node's neighbour array.
    nbs_array = data_graph[pitch_index]['nbs']
    # Find the index within node's neighbour array, else '-1'.
    next_pitch_index = next((i for i, item in enumerate(nbs_array) if item['nbs_pitch'] == next_pitch), -1)

    if next_pitch_index != -1:
        # Next pitch is already present in array, thus increase its count.
        nbs_array[next_pitch_index]['nbs_count'] += 1
    else:
        # Next pitch is not present in array, thus create and append a new neighbour node.
        nbs_array.append({
            'nbs_pitch': next_pitch,
            'nbs_count': 1
        })

print('Markov graph created.')

print('Predicting sequence...')

# Choose a random integer within the note's total count.
random_int = random.choice(range(1, len(notes)))

counter = 0
# Iterate through neighbour array until the neighbour occurrence is greater than the random integer.
for node in data_graph:
    counter += node['count']
    if counter >= random_int:
        # Append neighbour to data out.
        markov_out.append(node['pitch'])
        s.append(note.Note(node['pitch']))
        break

# Generate range of notes equal to defined sequence length.
for generated_note in range(OUTPUT_LENGTH - 1):

    # Get node's neighbour array and total occurrence.
    pitch_index = next((i for i, item in enumerate(data_graph) if item['pitch'] == markov_out[generated_note]), -1)
    nbs_array = data_graph[pitch_index]['nbs']
    total_count = data_graph[pitch_index]['count']

    # Choose a random integer within the note's total count.
    random_int = random.choice(range(1, total_count))

    counter = 0
    # Iterate through neighbour array until the neighbour occurrence is greater than the random integer.
    for neighbour in nbs_array:
        counter += neighbour['nbs_count']
        if counter >= random_int:
            # Append neighbour to data out.
            markov_out.append(neighbour['nbs_pitch'])
            break

print('Prediction complete.')

midi_generator.create_midi(markov_out, 'markov')
