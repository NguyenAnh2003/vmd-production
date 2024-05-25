import streamlit as st
import numpy as np
import soundfile as sf
import time
from st_audiorec import st_audiorec
from supabase import create_client, Client
import scipy.io.wavfile as wavfile
import os

# init DB
url: str = "https://yyciwuqbkcqecbrqholh.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inl5Y2l3dXFia2NxZWNicnFob2xoIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcxNjI5NTEwNSwiZXhwIjoyMDMxODcxMTA1fQ.5mRyn4e7g1PKBnh2N6g10ISkp7CvQnX2owbWQLe9lnQ"
DB: Client = create_client(supabase_url=url, supabase_key=key)


def colorize(value):
    if value == 1:
        return "color: green"
    elif value == 0:
        return "color: red"
    else:
        return ""


def main():
    # sample for select box
    _, cl1, _, cl3, _ = st.columns([1, 10, 1, 6, 1])
    with cl1:
        # setup interface
        st.markdown("<h1>Thu thập dữ liệu</h1>", unsafe_allow_html=True)
        st.markdown("<span style='color: red ;font-size: 20px'>Bạn vui lòng đọc hướng dẫn sử dụng</span>",
                    unsafe_allow_html=True)

        scol1, scol2 = st.columns([2, 2])

        #
        with scol1:
            canonical = st.text_input("Từ phát âm đúng")
            st.markdown(
                f"""<div style="display: flex; gap: 10px"><p style='font-size: 15px; color: 'black'>Từ phát âm đúng
                                    <span style='font-size: 20px; color: 'red'><strong>{canonical}</strong></span></p></div>""",
                unsafe_allow_html=True)

        with scol2:
            transcript = st.text_input("Từ phát âm")
            st.markdown(
                f"""<div style="display: flex; gap: 10px"><p style='font-size: 15px; color: 'black'>Từ phát âm
                                    <span style='font-size: 20px; color: 'red'><strong>{transcript}</strong></span></p></div>""",
                unsafe_allow_html=True)

        # RECORD AUDIO WITH STREAMLIT-AUDIOREC
        wav_audio_data = st_audiorec()

        # audio_bytes = audio_recorder(text="", pause_threshold=1, sample_rate=44100, energy_threshold=0.)

        # if audio_bytes:
        #     st.audio(audio_bytes, format="audio/wav")

        if st.button("Lưu dữ liệu") and wav_audio_data:
            if canonical != '' and transcript != '':
                # Convert audio_bytes to a NumPy array
                audio_array = np.frombuffer(wav_audio_data, dtype=np.int32)

                if len(audio_array) > 0:
                    # Save the audio to a file using soundfile library
                    # You can change the filename and format accordingly
                    OUT_WAV_FILE = f"./upload/recorded_audio{time.time()}.wav"  # define absolute path

                    # wavfile.write(OUT_WAV_FILE, 44100, audio_array)
                    sf.write(OUT_WAV_FILE, audio_array, 44100)

                    # send audio file
                    bucket_res = DB.storage.from_("data-collect-bucket").upload(file=OUT_WAV_FILE,
                                                                                path=f"{OUT_WAV_FILE}",
                                                                                file_options={
                                                                                    "content-type": "audio/wav"})
                    print(f"Bucket: {bucket_res}")
                    if OUT_WAV_FILE:
                        # get audio_url
                        wav_url = DB.storage.from_("data-collect-bucket").get_public_url(path=f"{OUT_WAV_FILE}")
                        print(f"Wav url: {wav_url}")
                        st.write("Đang chờ xử lý")

                        response = DB.table("speech-data").insert(
                            {"audio_url": wav_url, "canonical": canonical.strip(),
                             "transcripts": transcript.strip()}).execute()

                        print(f"DB: {response}")
                        wav_audio_data = None
                        if response:
                            st.markdown(f"<div style='color: Green; font-size: 25px'>Done</div>", unsafe_allow_html=True)

                            # delete wav file
                            if os.path.exists(OUT_WAV_FILE):
                                os.remove(OUT_WAV_FILE)
                        else:
                            st.error(f"Failed to fetch data")
                else:
                    st.warning("The audio data is empty.")
            else:
                st.warning("Điền đầy đủ thông tin bạn nhé")


if __name__ == "__main__":
    st.set_page_config(page_title="Collect Data", layout="wide")
    main()
