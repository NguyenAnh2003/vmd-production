FROM python:3.11-slim-bookworm

# caching req file when having changes
COPY requirements.txt .

# setup working dir
WORKDIR /vmd

# install requirements.txt
# copy all files to /vmd then should setup working dir
RUN pip install -r requirements.txt

# copy all files to vmd-app folder
# copy . . if using . /vmd it would be /vmd/vmd (not fk good)
COPY . .

# build multi-stages


# running docker container
# running server - using CMD not RUN when we had instance of image
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]