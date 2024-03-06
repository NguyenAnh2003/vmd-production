FROM python:3.11-slim-bookworm

# setup working dir
WORKDIR /vmd

# init venv & install requirements.txt
# avoid caching 
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install pydantic==2.5.3 fastapi==0.109.0 \
    python-multipart==0.0.6 torchaudio==2.1.2 \
    torch==2.1.2 numpy==1.25.2 python-dotenv==1.0.1 \
    uvicorn==0.27.1 transformers==4.38.1 librosa==0.10.1

# copy all files to vmd-app folder
# copy . . if using . /vmd it would be /vmd/vmd (not fk good)
COPY . .

# expose port
EXPOSE 8000

# running docker container
# running server - using CMD not RUN when we had instance of image
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]