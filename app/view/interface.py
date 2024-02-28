import streamlit as st
from audio_recorder_streamlit import audio_recorder
import io
import numpy as np
import soundfile as sf
import requests
import time
from dotenv import load_dotenv
import os

load_dotenv()

def colorize(value):
    if value == 1:
        return "color: green"
    elif value == 0:
        return "color: red"
    else:
        return ""

def main():
    # setup interface
    st.title(""" Hello, we need your information to improve our service """)

    text = st.text_input('Target text', '')
    st.write('The current test is', text)
    username = st.text_input("Your name", "")
    country = st.text_input("Your country", "")
    age = st.number_input("Your age", min_value=0)

    # Record audio using the audio_recorder function
    audio_bytes = audio_recorder(text="", pause_threshold=1, sample_rate=41_000)

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

    if st.button("Compute"):
        # Convert audio_bytes to a NumPy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

        if len(audio_array) > 0:
            # Save the audio to a file using soundfile library
            # You can change the filename and format accordingly
            """
            define temporarily upload dir (saving file from buffer)
            Read all byte in buffer after record and save the file
            Alternative can use Cloudinary service
            """
            OUT_WAV_FILE = f"upload/recorded_audio{time.time()}.wav" # define absolute path
            sf.write(OUT_WAV_FILE, audio_array, 41000, 'PCM_16')

            # Use the recorded audio directly for the API request
            file_data = {"file": (OUT_WAV_FILE,
                                  io.BytesIO(audio_bytes), "audio/wav")}
            # packaging data form
            data_package = {"text_target": text, "username": username, "country": country, "age": age}
            response = requests.post("http://127.0.0.1:8000/danangvsr/vmd", files=file_data, data=data_package)

            if response.status_code == 200:
                # Parse the JSON response
                result = response.json() # casts to JSON

                result_html = ""
                for key, value in result.items():
                    result_html += f"<p style='{colorize(value)}; margin-left: 5px; font-size: 25px'>{key}</p>"
                st.markdown(f"<div style='display: flex; flex-direction: row; gap: 0;'>{result_html}</div>",
                            unsafe_allow_html=True)

                # Display the data in your Streamlit app
                # st.write("API Response:", data, response.status_code) # including status code and data
            else:
                st.error(f"Failed to fetch data. Status code: {response.status_code}")
        else:
            st.warning("The audio data is empty.")

# run interface ui `streamlit run app/view/interface.py`
st.set_page_config(page_title="Mispronunciation detection")
main()