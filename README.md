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
streamlit run app/view/interface.py
```

### Model .pth and .env file
https://drive.google.com/drive/folders/1jrACQF0nceiSlgTyUmrRJSgX9VC0pd5O?usp=drive_link

## Attention when cloned repo 
1. create logs folder in core
2. create upload folder in root
3. create saved_model folder in root

## Docker build
```python
docker build -t vmd-api
```

## Docker run container
```python
docker run -p 8000:8000 vmd-api
```
