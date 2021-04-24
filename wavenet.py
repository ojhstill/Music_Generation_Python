import numpy as np
import keras.backend as k
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from collections import Counter
from sklearn.model_selection import train_test_split

import midi_reader
import midi_generator

FREQ_THRESHOLD = 25
TIME_STEPS = 100
OUTPUT_LENGTH = 64


def wavenet(unique_x, unique_y):
    # Embedding layer.
    model.add(Embedding(len(unique_x), 100, input_length=32, trainable=True))

    model.add(Conv1D(64, 3, padding='causal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPool1D(2))

    model.add(Conv1D(128, 3, activation='relu', dilation_rate=2, padding='causal'))
    model.add(Dropout(0.2))
    model.add(MaxPool1D(2))

    model.add(Conv1D(256, 3, activation='relu', dilation_rate=4, padding='causal'))
    model.add(Dropout(0.2))
    model.add(MaxPool1D(2))

    # model.add(Conv1D(256,5,activation='relu'))
    model.add(GlobalMaxPool1D())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(len(unique_y), activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    model.summary()

    return model


if __name__ == "__main__":

    # Get parsed midi data.
    notes_array = midi_reader.get_midi_dataset()

    # Convert 2D array into 1D array.
    notes = [element for note in notes_array for element in note]

    k.clear_session()
    model = Sequential()

    # Find required metadata.
    unique_notes = list(set(notes))
    n_vocab = len(set(notes))
    note_freq = dict(Counter(notes))

    # Only consider notes with an occurrence of more than defined threshold.
    frequent_notes = [note for note, count in note_freq.items() if count >= FREQ_THRESHOLD]

    new_music = []

    temp = []
    for note in notes:
        # for note in element:
        if note in frequent_notes:
            temp.append(note)
    new_music.append(temp)

    new_music = np.array(new_music)

    x = []
    y = []

    for note in new_music:
        for i in range(0, len(note) - TIME_STEPS, 1):
            input_seq = note[i:i + TIME_STEPS]
            output_seq = note[i + TIME_STEPS]

            x.append(input_seq)
            y.append(output_seq)

    x = np.array(x)
    y = np.array(y)

    unique_x = list(set(x.ravel()))
    x_note_to_int = dict((note, number) for number, note in enumerate(unique_x))

    x_seq = []
    for i in x:
        temp = []
        for j in i:
            temp.append(x_note_to_int[j])
        x_seq.append(temp)

    x_seq = np.array(x_seq)

    unique_y = list(set(y))
    y_note_to_int = dict((note, number) for number, note in enumerate(unique_y))
    y_seq = np.array([y_note_to_int[i] for i in y])

    x_tr, x_val, y_tr, y_val = train_test_split(x_seq, y_seq, test_size=0.2, random_state=0)

    model = wavenet(unique_x, unique_y)

    mc = ModelCheckpoint('wavenet_weights.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    history = model.fit(np.array(x_tr), np.array(y_tr), batch_size=128, epochs=50,
                        validation_data=(np.array(x_val), np.array(y_val)), verbose=1, callbacks=[mc])

    # Loading best model.
    from keras.models import load_model

    model = load_model('wavenet_weights.h5')

    import random

    ind = np.random.randint(0, len(x_val) - 1)

    random_music = x_val[ind]

    predictions = []
    for i in range(OUTPUT_LENGTH):
        random_music = random_music.reshape(1, TIME_STEPS)

        prob = model.predict(random_music)[0]
        y_pred = np.argmax(prob, axis=0)
        predictions.append(y_pred)

        random_music = np.insert(random_music[0], len(random_music[0]), y_pred)
        random_music = random_music[1:]

    print(predictions)

    x_int_to_note = dict((number, note) for number, note in enumerate(unique_x))
    predicted_notes = [x_int_to_note[i] for i in predictions]

    midi_generator.create_midi(predicted_notes, 'wavenet')
