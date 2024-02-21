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
import time
from dotenv import load_dotenv
import os

load_dotenv()
# run interface ui `streamlit run app/view/interface.py `

def main():
    # setup interface
    st.write(""" # Correction VMD
    Below having a input form and audio recorder, 
    the input form represent target text and audio record is the data want to detect ur mispronounce word 
    """)

    text = st.text_input('Target text', '')
    st.write('The current test is', text)

    # Record audio using the audio_recorder function
    audio_bytes = audio_recorder(pause_threshold=1, sample_rate=16_000)

    #
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
            sf.write(OUT_WAV_FILE, audio_array, 16000, 'PCM_16')

            # Use the recorded audio directly for the API request
            file_data = {"file": (OUT_WAV_FILE,
                                  io.BytesIO(audio_bytes), "audio/wav")}
            text_data = {"text_target": text}
            response = requests.post("http://localhost:8000/danangvsr/vmd", files=file_data,
                                     data=text_data,)

            if response.status_code == 200:
                # Parse the JSON response
                data = response.json() # casts to JSON

                # Display the data in your Streamlit app
                st.write("API Response:", data, response.status_code) # including status code and data
            else:
                st.error(f"Failed to fetch data. Status code: {response.status_code}")
        else:
            st.warning("The audio data is empty.")

if __name__ == "__main__":
    st.set_page_config(page_title="VMD DEMO")
    main()