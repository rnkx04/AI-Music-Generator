import os
import glob
import pickle
import numpy as np

from music21 import converter, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical


# =====================================================
# 1. EXTRACT NOTES FROM MIDI FILES
# =====================================================

def extract_notes(midi_folder="midi_data"):
    notes = []

    midi_files = (
        glob.glob(os.path.join(midi_folder, "*.mid"))
        + glob.glob(os.path.join(midi_folder, "*.midi"))
    )

    if not midi_files:
        raise FileNotFoundError(
            f"No MIDI files found in '{midi_folder}'."
        )

    print(f"Found {len(midi_files)} MIDI files.")

    for file_path in midi_files:

        print("Processing:", file_path)

        try:
            midi = converter.parse(file_path)

            # Read all notes and chords
            elements = midi.flatten().notes

            for element in elements:

                # Single note
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))

                # Chord
                elif isinstance(element, chord.Chord):
                    chord_notes = ".".join(
                        str(pitch)
                        for pitch in element.pitches
                    )

                    notes.append(chord_notes)

        except Exception as error:
            print("Error:", error)

    if not notes:
        raise ValueError(
            "No musical notes were extracted."
        )

    # Save extracted notes
    with open("notes.pkl", "wb") as file:
        pickle.dump(notes, file)

    print("Total musical tokens:", len(notes))
    print("Saved as notes.pkl")

    return notes


# =====================================================
# 2. PREPARE TRAINING SEQUENCES
# =====================================================

def prepare_sequences(notes, sequence_length=64):

    unique_notes = sorted(set(notes))

    vocabulary_size = len(unique_notes)

    print(
        "Unique notes/chords:",
        vocabulary_size
    )

    note_to_number = {
        note_name: number
        for number, note_name
        in enumerate(unique_notes)
    }

    number_to_note = {
        number: note_name
        for number, note_name
        in enumerate(unique_notes)
    }

    input_sequences = []
    output_notes = []

    for i in range(
        len(notes) - sequence_length
    ):

        sequence = notes[
            i:i + sequence_length
        ]

        target = notes[
            i + sequence_length
        ]

        input_sequences.append(
            [
                note_to_number[item]
                for item in sequence
            ]
        )

        output_notes.append(
            note_to_number[target]
        )

    if not input_sequences:
        raise ValueError(
            "Not enough MIDI data. "
            "Add more MIDI files or reduce "
            "sequence_length."
        )

    # Convert input to NumPy array
    input_sequences = np.array(
        input_sequences
    )

    # Normalize values
    input_sequences = (
        input_sequences.reshape(
            len(input_sequences),
            sequence_length,
            1
        )
        / float(vocabulary_size)
    )

    # Convert output to categorical data
    output_notes = to_categorical(
        output_notes,
        num_classes=vocabulary_size
    )

    print(
        "Training sequences:",
        len(input_sequences)
    )

    return (
        input_sequences,
        output_notes,
        vocabulary_size,
        number_to_note
    )


# =====================================================
# 3. CREATE LSTM MODEL
# =====================================================

def create_model(
    input_shape,
    vocabulary_size
):

    model = Sequential()

    model.add(
        LSTM(
            256,
            input_shape=input_shape,
            return_sequences=True
        )
    )

    model.add(
        Dropout(0.3)
    )

    model.add(
        LSTM(256)
    )

    model.add(
        Dropout(0.3)
    )

    model.add(
        Dense(
            128,
            activation="relu"
        )
    )

    model.add(
        Dropout(0.3)
    )

    model.add(
        Dense(
            vocabulary_size,
            activation="softmax"
        )
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam"
    )

    return model


# =====================================================
# 4. GENERATE NEW MUSIC
# =====================================================

def generate_music(
    model,
    input_sequences,
    number_to_note,
    vocabulary_size,
    number_of_notes=200
):

    # Select random starting sequence
    start_index = np.random.randint(
        0,
        len(input_sequences)
    )

    pattern = list(
        input_sequences[start_index]
        .flatten()
        .astype(int)
    )

    generated_notes = []

    print(
        f"Generating {number_of_notes} notes..."
    )

    for _ in range(number_of_notes):

        model_input = np.array(
            pattern
        ).reshape(
            1,
            len(pattern),
            1
        )

        model_input = (
            model_input
            / float(vocabulary_size)
        )

        prediction = model.predict(
            model_input,
            verbose=0
        )[0]

        # Choose the most likely note
        index = np.argmax(prediction)

        generated_notes.append(
            number_to_note[index]
        )

        # Move sequence forward
        pattern.append(index)
        pattern = pattern[1:]

    return generated_notes


# =====================================================
# 5. SAVE GENERATED MUSIC AS MIDI
# =====================================================

def save_midi(
    generated_notes,
    filename="generated_music.mid"
):

    music_stream = stream.Stream()

    offset = 0.0

    for item in generated_notes:

        # Chord
        if "." in item:

            pitches = item.split(".")

            chord_notes = []

            for pitch in pitches:

                try:
                    chord_notes.append(
                        note.Note(pitch)
                    )
                except Exception:
                    pass

            if chord_notes:

                new_chord = chord.Chord(
                    chord_notes
                )

                new_chord.offset = offset

                music_stream.append(
                    new_chord
                )

        # Single note
        else:

            try:

                new_note = note.Note(item)

                new_note.offset = offset

                music_stream.append(
                    new_note
                )

            except Exception:
                pass

        offset += 0.5

    music_stream.write(
        "midi",
        fp=filename
    )

    print(
        "Generated MIDI saved as:",
        filename
    )


# =====================================================
# 6. MAIN PROGRAM
# =====================================================

def main():

    MIDI_FOLDER = "midi_data"

    SEQUENCE_LENGTH = 64

    EPOCHS = 30

    BATCH_SIZE = 64

    GENERATED_NOTES = 200

    # Create MIDI folder
    os.makedirs(
        MIDI_FOLDER,
        exist_ok=True
    )

    # -----------------------------------------------
    # Load or extract notes
    # -----------------------------------------------

    if os.path.exists("notes.pkl"):

        print("Loading existing notes.pkl...")

        with open(
            "notes.pkl",
            "rb"
        ) as file:

            notes = pickle.load(file)

    else:

        notes = extract_notes(
            MIDI_FOLDER
        )

    # -----------------------------------------------
    # Prepare sequences
    # -----------------------------------------------

    (
        input_sequences,
        output_notes,
        vocabulary_size,
        number_to_note
    ) = prepare_sequences(
        notes,
        SEQUENCE_LENGTH
    )

    # -----------------------------------------------
    # Create model
    # -----------------------------------------------

    print("\nCreating LSTM model...")

    model = create_model(
        (
            input_sequences.shape[1],
            input_sequences.shape[2]
        ),
        vocabulary_size
    )

    model.summary()

    # -----------------------------------------------
    # Train model
    # -----------------------------------------------

    print("\nStarting training...\n")

    model.fit(
        input_sequences,
        output_notes,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # -----------------------------------------------
    # Generate music
    # -----------------------------------------------

    print("\nGenerating new music...\n")

    generated_notes = generate_music(
        model,
        input_sequences,
        number_to_note,
        vocabulary_size,
        GENERATED_NOTES
    )

    # -----------------------------------------------
    # Save MIDI
    # -----------------------------------------------

    save_midi(
        generated_notes,
        "generated_music.mid"
    )

    print("\n================================")
    print("AI MUSIC GENERATION COMPLETED!")
    print("================================")


# =====================================================
# START PROGRAM
# =====================================================

if __name__ == "__main__":
    main()
