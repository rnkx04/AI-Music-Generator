from music21 import converter, note, chord
import glob
import pickle

notes = []

# MIDI files read karna
for file in glob.glob("dataset/*.mid"):
    print("Processing:", file)

    midi = converter.parse(file)

    for element in midi.flatten().notes:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))

        elif isinstance(element, chord.Chord):
            notes.append(".".join(str(n) for n in element.normalOrder))

print("Total notes:", len(notes))

# Notes save karna
with open("notes.pkl", "wb") as file:
    pickle.dump(notes, file)

print("Preprocessing complete!")
print("Notes saved in notes.pkl")
