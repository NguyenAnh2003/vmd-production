FROM python:3.11-alpine

# setup working dir
WORKDIR /vmd

# caching req file when having changes
COPY requirements.txt .

# init venv & install requirements.txt
RUN python -m venv venv && \
    . venv/bin/activate && \ 
    pip install -r requirements.txt

# copy all files to vmd-app folder
# copy . . if using . /vmd it would be /vmd/vmd (not fk good)
COPY . .

# expose port
EXPOSE 8000

# running docker container
# running server - using CMD not RUN when we had instance of image
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]