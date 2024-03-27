import streamlit as st
import numpy as np
import soundfile as sf
import time
import supabase
from st_audiorec import st_audiorec
from supabase import create_client, Client
import scipy.io.wavfile as wavfile
import os
from pathlib import Path
import sys

# init DB
url: str = "https://cceebjjirmrvyhqecubk.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNjZWViamppcm1ydnlocWVjdWJrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDk2NDMxMTMsImV4cCI6MjAyNTIxOTExM30.dh4WE15QV41Ch7GZlpNyELOa6ZZiapV9RsYHuHi6ZQ8"
DB: Client = create_client(supabase_url=url, supabase_key=key)

# demo app using streamlit integrating model prediction -> return mapped result
# call api to save data recorded and call model api to predict

def colorize(value):
    if value == 1:
        return "color: green"
    elif value == 0:
        return "color: red"
    else:
        return ""

def _get_phonemes(file_path):
    list_of_phonemes = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            list_of_phonemes.append(line)

    return list_of_phonemes

def main():
    # sample for select box
    list_phonemes = _get_phonemes("phoneme_dict.txt")
    cl1, _, cl3  = st.columns([3, 1, 2])
    with cl1:
        # setup interface
        st.markdown("<h1>Thu thập dữ liệu</h1>", unsafe_allow_html=True)
        st.markdown("<span style='color: red ;font-size: 20px'>Bạn vui lòng đọc hướng dẫn sử dụng</span>", unsafe_allow_html=True)

        scol1, scol2 = st.columns([3, 2])

        with scol1:
            # toggle box
            suggestion = st.selectbox(
                "Gợi ý tự bạn muốn phát âm",
                list_phonemes,
                index=0,
                placeholder="Select contact method...",
            )
        with scol2:
            selected_suggetion = suggestion.split("-")
            if suggestion:
                target_text = st.text_input("Từ bạn muốn phát âm", selected_suggetion[0])
            else:
                target_text = st.text_input('(tối đa 2 từ E.g: vào nụi)', '')


        # text = st.text_input('(tối đa 2 từ E.g: vào nụi)', '')
        username = st.text_input("Tên của bạn", "")
        country = st.text_input("Quê quán (E.g: Đà Nằng or ĐN)", "")
        age = st.number_input("Tuổi", min_value=0)

        # Record audio using the audio_recorder function
        st.markdown(f"<p style='font-size: 15px; color: 'black'>Từ bạn muốn phát âm "
                    f"<span style='font-size: 20px; color: 'red'><strong>{target_text}</strong></span></p>", unsafe_allow_html=True)

        # RECORD AUDIO WITH STREAMLIT-AUDIOREC
        wav_audio_data = st_audiorec()

        # audio_bytes = audio_recorder(text="", pause_threshold=1, sample_rate=44100, energy_threshold=0.)

        # if audio_bytes:
        #     st.audio(audio_bytes, format="audio/wav")

        if st.button("Lưu dữ liệu") and wav_audio_data:
            if username != '' and target_text != '' and age != 0 and country != '':
                # Convert audio_bytes to a NumPy array
                audio_array = np.frombuffer(wav_audio_data, dtype=np.int32)

                if len(audio_array) > 0:
                    # Save the audio to a file using soundfile library
                    # You can change the filename and format accordingly
                    wavfile.write(f"upload/recorded_audio{time.time()}.wav", 44100, audio_array)

                    OUT_WAV_FILE = f"upload/recorded_audio{time.time()}.wav"  # define absolute path
                    sf.write(OUT_WAV_FILE, audio_array, 44100)

                    # send audio file
                    bucket_res = DB.storage.from_("vmd-bucket").upload(file=OUT_WAV_FILE, path=f"{OUT_WAV_FILE}",
                                                                       file_options={"content-type": "audio/wav"})
                    print(f"Bucket: {bucket_res}")
                    if OUT_WAV_FILE:
                        # get audio_url
                        wav_url = DB.storage.from_("vmd-bucket").get_public_url(path=f"{OUT_WAV_FILE}")
                        print(f"Wav url: {wav_url}")
                        st.write("Đang chờ xử lý")
                        response = DB.table("vmd-data").insert({"audio_url": wav_url, "text_target": target_text.strip(),
                                                                "username": username, "country": country,
                                                                "age": age}).execute()
                        print(f"DB: {response}")
                        if response:
                            st.markdown(f"<div style='color: red; font-size: 25px'>Cảm ơn bạn đã giành thời gian</div>",
                                        unsafe_allow_html=True)
                            
                            # delete wav file
                            if os.path.exists(OUT_WAV_FILE):
                                os.remove(OUT_WAV_FILE)
                        else:
                            st.error(f"Failed to fetch data")
                else:
                    st.warning("The audio data is empty.")
            else:
                st.title("Điền đầy đủ thông tin bạn nhé")

    with cl3:
        st.markdown(f"<h2>Hướng dẫn sử dụng</h2>", unsafe_allow_html=True)

        st.markdown(f"<p><strong>Bước 1</strong> Chọn từ bạn muốn ghi âm </br>"
                    f"Bạn có thể chọn từ trong hộp gợi ý hoặc từ ghi </br> "
                    f"<strong>(ghi xong nhấn enter giúp mình)</strong></p>", unsafe_allow_html=True)

        st.markdown(f"<p><strong>Bước 2</strong> Điền đẩy đủ thông tin</p>"
                    f"<p><strong>Bước 3</strong> Bấm vào Start Recording để thu âm giúp mình</p>"
                    f"<p><strong>Bước 4</strong> Ngoài việc phát âm đúng từ hiện tại </br> bạn có thể chọn phát âm sai như trong hộp gợi ý, </br> hoặc phát âm sai như ví dụ bên dưới </br>"
                    f"<strong><span style='color: red'>Nhóm chúng mình cần bạn phát âm một từ với </br> 4 audio (1 phát âm đúng và 3 phát âm sai).</span></strong> </br>"
                    f"<strong><span style='color: green'>Eg: vào nụi -> vào núi, vào nui, vào nùi</span></strong></p> </br>"
                    f"<strong><span style='color: red'>Lưu ý đợi thanh màu đỏ hiện lên rồi phát âm nhé</span></strong> </br>", unsafe_allow_html=True)
        st.image("visualize.png", width=300)
        st.markdown(f"<strong><span style='font-size: 25px'>Cảm ơn sự hợp tác của bạn rất nhiều</span></strong>", unsafe_allow_html=True)

if __name__ == "__main__":
    # run interface ui `streamlit run app/view/data_collection_ui.py`
    st.set_page_config(page_title="Mispronunciation detection", layout="wide")
    main()