import streamlit as st
from audio_recorder_streamlit import audio_recorder
import io
import numpy as np
import soundfile as sf
import requests
import time
from dotenv import load_dotenv
import supabase
from supabase import create_client, Client
import os

# init DB
url: str = "https://cceebjjirmrvyhqecubk.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNjZWViamppcm1ydnlocWVjdWJrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDk2NDMxMTMsImV4cCI6MjAyNTIxOTExM30.dh4WE15QV41Ch7GZlpNyELOa6ZZiapV9RsYHuHi6ZQ8"
DB: Client = create_client(supabase_url=url, supabase_key=key)

print(f"Supabase: {DB}")

# demo app using streamlit integrating model prediction -> return mapped result
# call api to save data recorded and call model api to predict

def colorize(value):
    if value == 1:
        return "color: green"
    elif value == 0:
        return "color: red"
    else:
        return ""


def main():
    # setup interface
    st.title(""" Data collection """)

    text = st.text_input('Target text', '')
    st.write('The current test is', text)
    username = st.text_input("Your name", "")
    country = st.text_input("Your Residence (E.g: Đà Nằng or ĐN)", "")
    age = st.number_input("Your age", min_value=0)

    # Record audio using the audio_recorder function
    col1, col2, _, _ = st.columns(4)
    with col1:
        audio_bytes = audio_recorder(text="", pause_threshold=1, sample_rate=44100, energy_threshold=0.)
    with col2:
        if st.button("Reload"):
            audio_bytes = []

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

    if st.button("Save data"):
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
            OUT_WAV_FILE = f"upload/recorded_audio{time.time()}.wav"  # define absolute path
            sf.write(OUT_WAV_FILE, audio_array, 44100, 'PCM_24')

            # send audio file
            bucket_res = DB.storage.from_("vmd-bucket").upload(file=OUT_WAV_FILE, path=f"{OUT_WAV_FILE}",
                                                  file_options={"content-type": "audio/wav"})
            print(f"Bucket: {bucket_res}")
            if OUT_WAV_FILE:
                # get audio_url
                wav_url = DB.storage.from_("vmd-bucket").get_public_url(path=f"{OUT_WAV_FILE}")
                print(f"Wav url: {wav_url}")
                response = DB.table("vmd-data").insert({"audio_url": wav_url, "text_target": text,
                                                        "username": username, "country": country,
                                                        "age": age}).execute()
                print(f"DB: {response}")

                if response:
                    st.write("THANKS = ]]]")

                else:
                    st.error(f"Failed to fetch data")
        else:
            st.warning("The audio data is empty.")


# run interface ui `streamlit run app/view/data_collection_interface.py`
st.set_page_config(page_title="Mispronunciation detection")
main()