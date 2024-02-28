import streamlit as st

def colorize(value):
    if value == 1:
        return "color: green"
    elif value == 0:
        return "color: red"
    else:
        return ""

def main():
    result = {"bao": 1, "vây": 0, "anh": 1, "bảy": 0}
    result_html = ""
    for key, value in result.items():
        result_html += f"<p style='{colorize(value)}; margin-left: 5px; font-size: 25px'>{key}</p>"
    st.markdown(f"<div style='display: flex; flex-direction: row; gap: 0;'>{result_html}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
