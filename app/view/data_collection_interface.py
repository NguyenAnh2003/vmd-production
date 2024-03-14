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
import scipy.io.wavfile as wavfile
import os
import wave

# init DB
url: str = "https://cceebjjirmrvyhqecubk.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNjZWViamppcm1ydnlocWVjdWJrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDk2NDMxMTMsImV4cCI6MjAyNTIxOTExM30.dh4WE15QV41Ch7GZlpNyELOa6ZZiapV9RsYHuHi6ZQ8"
DB: Client = create_client(supabase_url=url, supabase_key=key)

print(f"Supabase: {DB}")

# demo app using streamlit integrating model prediction -> return mapped result
# call api to save data recorded and call model api to predict

pronounce_words = ["vào nụi", "bao vây", "anh bảy"]

def colorize(value):
    if value == 1:
        return "color: green"
    elif value == 0:
        return "color: red"
    else:
        return ""


def main():
    cl1, _, cl3  = st.columns([3, 1, 2])
    with cl1:
        # setup interface
        st.markdown("<h1>Thu thập dữ liệu</h1>", unsafe_allow_html=True)
        st.markdown("<span style='color: red ;font-size: 20px'>Bình tĩnh đọc hướng dẫn sử dụng đã</span>", unsafe_allow_html=True)


        # toggle box
        text = st.selectbox(
            "Từ bạn muốn phát âm",
            pronounce_words,
            index=0,
            placeholder="Select contact method...",
        )
        # text = st.text_input('(tối đa 2 từ E.g: vào nụi)', '')
        username = st.text_input("Tên của bạn", "")
        country = st.text_input("Quê quán (E.g: Đà Nằng or ĐN)", "")
        age = st.number_input("Tuổi", min_value=0)

        # Record audio using the audio_recorder function
        col1, col2, _, _ = st.columns([4, 2, 1, 1])
        with col1:
            st.markdown(f"<p style='font-size: 15px; color: red'>Phát âm theo từ "
                        f"<strong>{text}</strong></p>", unsafe_allow_html=True)
            audio_bytes = audio_recorder(text="", pause_threshold=1, sample_rate=44100, energy_threshold=0.)
        with col2:
            if st.button("Tải lại"):
                audio_bytes = []

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")

        if st.button("Lưu dữ liệu"):
            if username != '' and text != '' and age != 0 and country != '':
                # Convert audio_bytes to a NumPy array
                print(audio_bytes)
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

                if len(audio_array) > 0:
                    # Save the audio to a file using soundfile library
                    # You can change the filename and format accordingly
                    """
                    define temporarily upload dir (saving file from buffer)
                    Read all byte in buffer after record and save the file
                    Alternative can use Cloudinary service
                    """
                    wavfile.write(f"upload/recorded_audio{time.time()}.wav", 44100, audio_array)

                    # OUT_WAV_FILE = f"upload/recorded_audio{time.time()}.wav"  # define absolute path
                    # sf.write(OUT_WAV_FILE, audio_array, 44100, 'PCM_24')

                    # send audio file
                    # bucket_res = DB.storage.from_("vmd-bucket").upload(file=OUT_WAV_FILE, path=f"{OUT_WAV_FILE}",
                    #                                                    file_options={"content-type": "audio/wav"})
                    # print(f"Bucket: {bucket_res}")
                    # if OUT_WAV_FILE:
                    #     # get audio_url
                    #     wav_url = DB.storage.from_("vmd-bucket").get_public_url(path=f"{OUT_WAV_FILE}")
                    #     print(f"Wav url: {wav_url}")
                    #     response = DB.table("vmd-data").insert({"audio_url": wav_url, "text_target": text,
                    #                                             "username": username, "country": country,
                    #                                             "age": age}).execute()
                    #     print(f"DB: {response}")
                    #
                    #     if response:
                    #         st.write("Thanks")
                    #
                    #     else:
                    #         st.error(f"Failed to fetch data")
                else:
                    st.warning("The audio data is empty.")
            else:
                st.title("Điền đầy đủ thông tin bạn nhé")

    with cl3:
        st.markdown(f"<h2>Hướng dẫn sử dụng</h2>", unsafe_allow_html=True)

        st.markdown(f"<p><strong>Bước 1</strong> Chọn từ bạn muốn ghi âm </p>", unsafe_allow_html=True)

        st.markdown(f"<p><strong>Bước 2</strong> Điền đẩy đủ thông tin</p>"
                    f"<p><strong>Bước 3</strong> Bấm vào Micro để thu âm giúp mình</p>"
                    f"<strong><span style='color: red'>Lưu ý bạn tạm nghỉ khoảng 1s rồi phát âm nhé</span></strong>"
                    f"<p><strong>Bước 4</strong> Ngoài việc phát âm đúng từ hiện tại </br> làm ơn phát âm biến thể của từ đó</p>", unsafe_allow_html=True)
        st.write("")

if __name__ == "__main__":
    # run interface ui `streamlit run app/view/data_collection_interface.py`
    st.set_page_config(page_title="Mispronunciation detection", layout="wide")
    main()