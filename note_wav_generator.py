import numpy as np
import soundfile as sf  # Library for audio file I/O

# Set the sampling rate and duration
sr = 44100  # Sampling rate (samples per second)
duration = 3  # Duration in seconds

# Frequency of the note A4 (440 Hz)
frequency = 445.0

# Generate a sine wave signal for A4 note
t = np.linspace(0, duration, int(sr * duration), endpoint=False)  # Time array
note = np.sin(2 * np.pi * frequency * t)  # Sine wave for A4 note

# Save the generated sine wave to an audio file
sf.write('wav/sharp2_A4.wav', note, sr)
