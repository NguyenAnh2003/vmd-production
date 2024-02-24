import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

def record_audio(duration, fs):
    st.text("Recording...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    st.text("Recording done.")
    return audio_data.flatten()

def main():
    st.title("Streamlit Audio Recorder")

    duration = st.slider("Recording Duration (seconds):", 1, 10, 5)
    fs = 44100  # You can adjust the sample rate if needed

    if st.button("Record Audio"):
        audio_data = record_audio(duration, fs)

        # Save the recorded audio as a WAV file
        write("recorded_audio.wav", fs, audio_data)

        st.audio("recorded_audio.wav", format="audio/wav", start_time=0)

if __name__ == "__main__":
    main()
