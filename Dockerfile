# 
FROM python:3.10-slim

# setup working dir
WORKDIR /vmd-app

# copy all files to vmd-app folder
# copy . . if using . /vmd-app it would be /vmd-app/vmd-app (not fk good)
COPY . .

# install requirements.txt
# copy all files to /vmd-app then should setup working dir
RUN pip install -r requirements.txt

# set port for container
EXPOSE 80

# running docker container
# running server - using CMD not RUN when we had instance of image
CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"]

# docker build 
# docker build -t vmd-api .

# docker run
# docker run -p 8080:80 vmd-api