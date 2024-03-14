import streamlit as st
import sounddevice as sd
import numpy as np
import soundfile as sf
import time


def record_audio(duration, samplerate=44100, channels=2):
    st.warning("Recording started. Speak into the microphone...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
    sd.wait()
    st.warning("Recording stopped.")
    return audio_data, samplerate


def main():
    st.title("Audio Recorder")

    recording = False

    if st.button("Start Recording"):
        recording = True
        start_time = time.time()  # Start time of recording

    if recording:
        # Recording is active
        elapsed_time = time.time() - start_time
        st.write(f"Recording Duration: {elapsed_time:.2f} seconds")
        audio_data, samplerate = record_audio(duration=elapsed_time)

        if st.button("Stop Recording"):
            recording = False
            # Save audio to WAV file
            save_path = "recorded_audio.wav"
            sf.write(save_path, audio_data, samplerate)
            st.success(f"Audio saved as {save_path}")


if __name__ == "__main__":
    main()
