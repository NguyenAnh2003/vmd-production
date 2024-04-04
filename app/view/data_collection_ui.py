import streamlit as st
import numpy as np
import soundfile as sf
import time
import supabase
from st_audiorec import st_audiorec
from supabase import create_client, Client
import scipy.io.wavfile as wavfile
from random import shuffle
import os
import requests
from pathlib import Path
import sys
from PIL import Image

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
    list_phonemes = shuffle(_get_phonemes("phoneme_dict.txt"))
    cl1, _, cl3 = st.columns([3, 1, 2])
    with cl1:
        # setup interface
        st.markdown("<h1>Thu thập dữ liệu</h1>", unsafe_allow_html=True)
        st.markdown("<span style='color: red ;font-size: 20px'>Bạn vui lòng đọc hướng dẫn sử dụng</span>",
                    unsafe_allow_html=True)

        scol1, scol2, scol3 = st.columns([3, 2, 2])

        with scol1:
            # toggle box
            suggestion = st.selectbox(
                "Gợi ý tự bạn muốn phát âm (phát âm đúng - sai)",
                list_phonemes,
                index=0,
                placeholder="Select contact method...",
            )
        # 
        selected_suggetion = suggestion.split("-")

        # 
        with scol2:
            if suggestion:
                target_text = st.text_input("Từ phát âm đúng", selected_suggetion[0])
            else:
                target_text = st.text_input('(tối đa 2 từ E.g: vào nụi)', '')

        # 
        with scol3:
            if suggestion:
                m_words = selected_suggetion[1].split(",")
                mispronouned_word = st.text_input("Từ bạn muốn phát âm", m_words[0])
            else:
                mispronouned_word = st.text_input(f"Phát âm sai của f{target_text}", "")

        # text = st.text_input('(tối đa 2 từ E.g: vào nụi)', '')
        username = st.text_input("Tên của bạn", "")
        country = st.text_input("Quê quán (E.g: Đà Nằng or ĐN)", "")
        age = st.number_input("Tuổi", min_value=0)

        # Record audio using the audio_recorder function

        st.markdown(
            """<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min
            .css">"""
            f"""<div style="display: flex; gap: 10px"><p style='font-size: 15px; color: 'black'>Từ phát âm đúng 
            <span style='font-size: 20px; color: 'red'><strong>{target_text}</strong></span></p></div>""",
            unsafe_allow_html=True)

        sscol1, sscol2, sscol3 = st.columns([1, 1, 1])

        with sscol1:
            st.markdown(
                f"<p>Từ bạn muốn phát âm <span style='font-size: 20px; color: 'red'><strong>{mispronouned_word}</strong"
                f"></span></p>",
            unsafe_allow_html=True)

        with sscol2:
            if st.button("Nghe từ muốn phát âm"):
                url = 'https://api.fpt.ai/hmi/tts/v5'

                payload = mispronouned_word
                headers = {
                    'api-key': '03Aw9xRXvspjlbUTlpJway0DTznJ01HY',
                    'speed': '-2.5',
                    'voice': 'banmai'
                }

                response = requests.request('POST', url, data=payload.encode('utf-8'), headers=headers)

                audio_url = response.text.split("\"")[3]

                st.audio(audio_url, format='audio/wav', start_time=0)

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
                        response = DB.table("vmd-data").insert(
                            {"audio_url": wav_url, "canonical_text": target_text.strip(),
                             "transcript_text": mispronouned_word.strip(),
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

        st.markdown(f"<p><strong>Bước 1</strong> Chọn từ bạn muốn ghi âm, chọn từ trong hộp gợi ý hoặc tự chọn.</p>"
                    f"<p><strong>Bước 2</strong> <strong>Điền đầy đủ thông tin </strong>, đặc biệt là ”<strong>từ "
                    f"phát âm đúng”</strong> và”<strong>từ bạn muốn phát âm</strong>”. Lưu ý <strong>từ muốn bạn "
                    f"phát âm</strong> là <strong>từ bạn sẽ phát âm khi ghi âm.</strong> Bạn có thể nghe thử cách "
                    f"phát âm ở bên cạnh.</p>"
                    f"<p><strong>Bước 3</strong> Bấm <strong>“Start Recording”</strong> để thu âm, sau khi thu âm "
                    f"xong bấm ”<strong>Stop</strong>” và nghe lại phần ghi âm ở bên dưới. Nếu phần ghi âm <strong>bị "
                    f"lỗi hoặc thiếu </strong>thì bấm <strong>“Reset”</strong> để ghi âm lại nha.</p>"
                    f"<p><strong>Bước 4</strong> Bấm <strong>“Lưu dữ liệu”</strong> để gửi ghi âm về cho chúng mình "
                    f"bạn nhé</p></br>"
                    f"<strong><span style='color: red'>Lưu ý: </span></strong> Nhóm chúng mình cần dữ liệu phát âm "
                    f"sai, bạn có thể giúp chúng mình phát âm <strong>1 từ với 4 bản ghi âm: 1 bản phát âm đúng và 3 "
                    f"bản phát âm sai.</strong> Ví dụ: <strong>“vào núi (phát âm đúng) - vào nui, vào nùi, "
                    f"vào nụi (phát âm sai).</strong></br>"
                    f"<strong><span style='color: green'>Eg: vào nụi(phát âm đúng) -> vào núi, vào nui, vào nùi(phát "
                    f"âm sai)</span></strong></p> </br>"
                    f"<strong><span style='color: red'>Khi thanh ghi âm hiện lên/sáng lên bạn hẳn phát âm "
                    f"nhé.</span></strong> </br>",
                    unsafe_allow_html=True)
        st.image("visualize.png", width=300)  # aaaa
        st.markdown(f"<strong><span style='font-size: 25px'>Cảm ơn sự giúp đỡ của bạn rất nhiều</span></strong>",
                    unsafe_allow_html=True)


if __name__ == "__main__":
    st.set_page_config(page_title="Mispronunciation detection", layout="wide")
    main()
