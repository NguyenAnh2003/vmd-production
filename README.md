### Init venv
```python
python -m venv venv
```

### Activate venv to download dependencies
```python 
Window: .\venv\Scripts\activate
Linux: source venv/bin/activate
```

### Install dependencies
```python
pip install -r requirements.txt
```

### Run streamlit
```python
streamlit run app/view/inteface.py
```

### Model .pth and .env file
https://drive.google.com/drive/folders/1jrACQF0nceiSlgTyUmrRJSgX9VC0pd5O?usp=drive_link

### Project Structure
```bash
├── README.md
├── app
│   ├── __pycache__
│   ├── exceptions.py
│   ├── routes.py
│   ├── services.py
│   └── view
├── configs
│   ├── linguistic_param.yaml
│   ├── phonetic_param.yaml
│   └── training_param.yaml
├── core
│   ├── logs
│   ├── mylogger.py
│   └── note.txt
├── feats
│   └── phonetic_embedding.py
├── main.py
├── model
│   ├── cnn_stack.py
│   ├── customize_model.py
│   ├── edit_distance.py
│   ├── metric.py
│   ├── rnn_stack.py
│   └── vmd_model.py
├── requirements.txt
├── saved_model
│   └── model_Customize_All_3e3.pth
├── setup
│   └── setup_app.py
├── test
│   ├── test.py
│   ├── test_inference.py
│   └── vao-nui_1618754929889.wav
├── test_main.http
├── upload
└── utils
    ├── char2phome.json
    ├── constants.py
    ├── dataset
    ├── translate.py
    └── utils.py
├── .env
```
#### Attention when cloned repo 
1. create logs folder in core
2. create upload folder in root
3. create saved_model folder in root
