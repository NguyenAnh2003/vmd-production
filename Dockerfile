# using pytorch base image CUDA 11.8 driver
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# setup working dir
WORKDIR /vmd

# cachine requirement
COPY requirements.txt .

# init venv & install requirements.txt
# avoid caching 
RUN apt-get install && apt-get update && \ 
    apt-get -y update && \
    python -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

# copy all files to vmd-app folder
# copy . . if using . /vmd it would be /vmd/vmd (not fk good)
COPY . .

# expose port
EXPOSE 8000

# running docker container
# running server - using CMD not RUN when we had instance of image
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
