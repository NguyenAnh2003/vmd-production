# 
FROM python:3.11-slim

# setup working dir
WORKDIR /vmd-app

# caching req file when having changes
COPY requirements.txt .

# install requirements.txt
# copy all files to /vmd-app then should setup working dir
RUN pip install -r requirements.txt

# copy all files to vmd-app folder
# copy . . if using . /vmd-app it would be /vmd-app/vmd-app (not fk good)
COPY . .

# running docker container
# running server - using CMD not RUN when we had instance of image
CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"]
