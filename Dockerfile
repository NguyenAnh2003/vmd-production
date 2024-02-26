# 
FROM python:3.10


# adding files to image
COPY . .

# setup working dir
WORKDIR /

# install requirements.txt
RUN pip install -r requirements.txt

# make port 80 vailable ?
EXPOSE 80


# running docker container
# running server
CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"]

# docker build 
# docker build -t vmd-api .

# docker run
# docker run -p 8080:80 vmd-api