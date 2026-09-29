import os
import urllib.request

os.makedirs("midi_data", exist_ok=True)

urls = [
    "https://www.piano-midi.de/midis/chopin/chpn-p1.mid",
    "https://www.piano-midi.de/midis/chopin/chpn-p2.mid",
    "https://www.piano-midi.de/midis/chopin/chpn-p3.mid",
    "https://www.piano-midi.de/midis/beethoven/beeth985-1.mid",
    "https://www.piano-midi.de/midis/mozart/mz_311_1.mid",
    "https://www.piano-midi.de/midis/bach/bach_846.mid",
]

headers = {'User-Agent': 'Mozilla/5.0'}

print("MIDI files download ho rahi hain...")
for url in urls:
    name = os.path.join("midi_data", url.split("/")[-1])
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp, open(name, "wb") as f:
            f.write(resp.read())
        print(f"Downloaded: {name}")
    except Exception as e:
        print(f"Error {url}: {e}")

print("Done! Saari files download ho gayi.")
