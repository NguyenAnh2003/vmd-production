from st_audiorec import st_audiorec
import streamlit as st
import soundfile
import numpy as np
import scipy.io.wavfile as wavfile
import time

wav_audio_data = st_audiorec()
OUT_WAV_FILE = f"upload/recorded_audio{time.time()}.wav"

audio_array = np.array([])

if wav_audio_data is not None:
    # process audio
    audio_array = np.frombuffer(wav_audio_data, dtype=np.int32)


if st.button("SAVE DATA"):
    soundfile.write(OUT_WAV_FILE, audio_array, 44100)