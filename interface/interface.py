"""
interface to report the research result
the interface will manipulate the streamlit library
"""
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import io
import numpy as np
import soundfile as sf
import requests

# Record audio using the audio_recorder function
audio_bytes = audio_recorder(pause_threshold=10.0, sample_rate=41_000)

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

if st.button("Save Audio"):
    # Convert audio_bytes to a NumPy array
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

    if len(audio_array) > 0:
        # Save the audio to a file using soundfile library
        # You can change the filename and format accordingly
        filename = "recorded_audio.wav"
        sf.write(filename, audio_array, 41000, 'PCM_16')
        st.success(f"Audio saved to {filename}")

        # Use the recorded audio directly for the API request
        file_data = {"file": ("recorded_audio.wav",
                              io.BytesIO(audio_bytes), "audio/wav")}
        text_data = {"text_target": "Your text data"}
        api_url = "http://localhost:8000/danangvsr/vmd"
        response = requests.post(api_url, files=file_data, data=text_data)

        # Check the response content and status code
        st.write("Response Status Code:", response.status_code)
        st.write("Response Content:", response.content)

        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()

            # Display the data in your Streamlit app
            st.write("API Response:", data)
        else:
            st.error(f"Failed to fetch data. Status code: {response.status_code}")
    else:
        st.warning("The audio data is empty.")
