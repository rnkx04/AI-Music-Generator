import os
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

def extract_notes_from_midi(midi_folder="midi_data"):
    notes = []
    midi_files = glob.glob(os.path.join(midi_folder, "*.mid")) + glob.glob(os.path.join(midi_folder, "*.midi"))
    
    if not midi_files:
        raise FileNotFoundError(f"'{midi_folder}/' folder me .mid files daalo pehle.")

    print(f"{len(midi_files)} MIDI files parse ho rahi hain...")

    for file_path in midi_files:
        try:
            midi = converter.parse(file_path)
            parts = instrument.partitionByInstrument(midi)
            notes_to_parse = parts.parts[0].flatten().notes if parts else midi.flatten().notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(p) for p in element.pitches))
        except Exception:
            continue

    with open('notes.pkl', 'wb') as f:
        pickle.dump(notes, f)
    
    return notes

def prepare_sequences(notes, sequence_length=64):
    pitchnames = sorted(list(set(notes)))
    n_vocab = len(pitchnames)

    note_to_int = {token: num for num, token in enumerate(pitchnames)}
    int_to_note = {num: token for num, token in enumerate(pitchnames)}

    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[token] for token in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    if n_patterns == 0:
        raise ValueError("Notes kam hain. Aur MIDI files daalo.")

    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1)) / float(n_vocab)
    categorical_output = to_categorical(network_output, num_classes=n_vocab)

    return network_input, normalized_input, categorical_output, n_vocab, int_to_note

def build_lstm_model(input_shape, n_vocab):
    model = Sequential([
        LSTM(256, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(256, return_sequences=False),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(n_vocab, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def sample_with_temperature(probabilities, temperature=0.8):
    probabilities = np.asarray(probabilities).astype('float64')
    log_probs = np.log(probabilities + 1e-10) / max(temperature, 1e-4)
    exp_probs = np.exp(log_probs)
    normalized_probs = exp_probs / np.sum(exp_probs)
    return np.random.choice(len(normalized_probs), p=normalized_probs)

def generate_notes(model, network_input, int_to_note, n_vocab, num_generate=200, temp=0.8):
    start_idx = np.random.randint(0, len(network_input) - 1)
    pattern = list(network_input[start_idx])
    prediction_output = []

    print(f"{num_generate} notes generate ho rahi hain...")

    for _ in range(num_generate):
        model_input = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        prediction = model.predict(model_input, verbose=0)[0]

        index = sample_with_temperature(prediction, temperature=temp)
        token = int_to_note[index]
        prediction_output.append(token)

        pattern.append(index)
        pattern = pattern[1:]

    return prediction_output

def save_to_midi(prediction_output, filename="output_composition.mid", offset_step=0.5):
    offset = 0.0
    output_elements = []

    for pattern in prediction_output:
        if '.' in pattern:
            chord_pitches = pattern.split('.')
            chord_notes = [note.Note(p) for p in chord_pitches]
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_elements.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            output_elements.append(new_note)

        offset += offset_step

    midi_stream = stream.Stream(output_elements)
    midi_stream.write('midi', fp=filename)
    print(f"MIDI file ready: {filename}")

if __name__ == "__main__":
    MIDI_FOLDER = "midi_data"
    os.makedirs(MIDI_FOLDER, exist_ok=True)

    if os.path.exists("notes.pkl"):
        with open("notes.pkl", "rb") as f:
            notes = pickle.load(f)
    else:
        notes = extract_notes_from_midi(MIDI_FOLDER)

    SEQ_LENGTH = 64
    net_in, norm_in, cat_out, n_vocab, int_to_note = prepare_sequences(notes, SEQ_LENGTH)

    model = build_lstm_model((norm_in.shape[1], norm_in.shape[2]), n_vocab)
    model.summary()

    checkpoint = ModelCheckpoint(
        "best_weights.weights.h5",
        monitor="loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    print("Training shuru...")
    model.fit(
        norm_in,
        cat_out,
        epochs=30,
        batch_size=64,
        callbacks=[checkpoint]
    )

    generated = generate_notes(model, net_in, int_to_note, n_vocab, num_generate=200, temp=0.8)
    save_to_midi(generated, filename="output_composition.mid")
